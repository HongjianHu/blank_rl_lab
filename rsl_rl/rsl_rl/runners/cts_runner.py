from __future__ import annotations

import copy

import torch
import os
import time

from tensordict import TensorDict

from rsl_rl.algorithms import CTS, MoENGCTS
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCriticCTS, ActorCriticMoENGCTS
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.storage import RolloutStorageCTS


class OnPolicyRunnerCTS(OnPolicyRunner):
    """Runner for concurrent teacher-student training."""

    alg: CTS

    def __init__(self, env, train_cfg, log_dir = None, device = "cpu"):
        super().__init__(env=env, train_cfg=train_cfg, log_dir=log_dir, device=device)

        self.history_length = self.alg.policy.history_length
        self.num_actor_obs = self.alg.policy.num_actor_obs

        self.history = torch.zeros(self.env.num_envs, self.history_length, self.num_actor_obs, device=device)

    def _get_policy_obs(
            self,
            obs: TensorDict,
        ) -> torch.Tensor:
        return self.alg.policy.get_actor_obs(obs)

    def _initialize_history(
            self,
            obs: TensorDict,
        ) -> None:
        self.history.zero_()
        current_policy_obs = self._get_policy_obs(obs)
        self.history[:, -1].copy_(current_policy_obs)

    def _update_history(
            self,
            obs: TensorDict,
            dones: torch.Tensor,
        ) -> None:
        done_mask = dones.reshape(-1).bool()
        self.history[done_mask] = 0.0

        current_policy_obs = self._get_policy_obs(obs)

        self.history = torch.cat(
            (
                self.history[:, 1:],
                current_policy_obs.unsqueeze(1),
            ),
            dim=1,
            )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False,) -> None:
        # 在训练刚开始时把每个环境的计数随机放到:[0,1249]
        # episode_length_s = 25.0, sim.dt = 0.005
        # episode_length_s / sim.dt = 1250
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, # shape:[num_envs],记录是目前episode运行的step
                high=int(self.env.max_episode_length),
            )

        self.logger.init_logging_writer()

        obs = self.env.get_observations().to(self.device)

        self._initialize_history(obs)
        self.train_mode()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            collect_start = time.time()

            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs, self.history)

                    next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    self.alg.process_env_step(
                        next_obs,
                        rewards,
                        dones,
                        extras,
                    )

                    self.logger.process_env_step(
                        rewards,
                        dones,
                        extras,
                        intrinsic_rewards=None,
                    )

                    self._update_history(next_obs, dones)
                    obs = next_obs
                collect_time = time.time() - collect_start

                self.alg.compute_returns(
                    obs,
                    self.history,
                )

            learn_start = time.time()
            loss_dict = self.alg.update()
            learn_time = time.time() - learn_start

            # Store the next iteration to run so save/load does not repeat the
            # iteration that has just completed.
            self.current_learning_iteration = it + 1

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.policy.action_std,
                rnd_weight=None,
            )

            if (self.logger.log_dir is not None and it % self.cfg["save_interval"] == 0):
                self.save(
                    os.path.join(
                        self.logger.log_dir,
                        f"model_{it}.pt",
                    )
                )

        if (self.logger.log_dir is not None and not self.logger.disable_logs):
            self.save(
                os.path.join(
                    self.logger.log_dir,
                    f"model_{self.current_learning_iteration}.pt",
                )
            )

    def save(self, path: str, infos: dict | None = None,) -> None:
        checkpoint = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "student_optimizer_state_dict":
                self.alg.student_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }

        torch.save(checkpoint, path)
        self.logger.save_model(
            path,
            self.current_learning_iteration,
        )

    def load(self, path:str, load_optimizer:bool = False, map_location:str | None = None) -> dict | None:
        # weights_only是否只加载张量权重，禁止反序列化任意 Python 对象
        # map_location指定加载后的张量应该放在哪个设备上
        checkpoint = torch.load(path, weights_only=False, map_location=map_location)

        resumed_training = self.alg.policy.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

            if "student_optimizer_state_dict" not in checkpoint:
                raise KeyError(
                    "CTS checkpoint does not contain "
                    "'student_optimizer_state_dict'."
                )

            self.alg.student_optimizer.load_state_dict(
                checkpoint["student_optimizer_state_dict"]
            )

            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]

            self.alg.student_encoder_learning_rate = self.alg.student_optimizer.param_groups[0]["lr"]

        if resumed_training:
            self.current_learning_iteration = checkpoint["iter"]

        # 重新启动训练程序后，IsaacLab 会创建新的环境
        self.history.zero_()
        self.alg.policy.reset()

        return checkpoint.get("infos")

    def _construct_algorithm(self, obs) -> CTS:
        if self.is_distributed:
            raise NotImplementedError(
                "Distributed CTS training is not implemented yet."
            )

        policy_cfg = copy.deepcopy(self.policy_cfg)
        algorithm_cfg = copy.deepcopy(self.alg_cfg)

        policy_class_name = policy_cfg.pop("class_name")
        algorithm_class_name = algorithm_cfg.pop("class_name")

        policy_classes: dict[str, type[ActorCriticCTS]] = {
            "ActorCriticCTS": ActorCriticCTS,
            "ActorCriticMoENGCTS": ActorCriticMoENGCTS,
        }

        algorithm_classes: dict[str, type[CTS]] = {
            "CTS": CTS,
            "MoENGCTS": MoENGCTS,
        }
        if policy_class_name not in policy_classes:
            raise ValueError(
                "OnPolicyRunnerCTS does not support policy class "
                f"{policy_class_name!r}."
            )

        if algorithm_class_name not in algorithm_classes:
            raise ValueError(
                "OnPolicyRunnerCTS does not support algorithm class "
                f"{algorithm_class_name!r}."
        )

        valid_pairs = {
            ("ActorCriticCTS", "CTS"),
            ("ActorCriticMoENGCTS", "MoENGCTS"),
        }

        selected_pair = (
            policy_class_name,
            algorithm_class_name,
        )

        if selected_pair not in valid_pairs:
            raise ValueError(
                "Incompatible CTS policy/algorithm pair: "
                f"{policy_class_name!r} with "
                f"{algorithm_class_name!r}."
        )

        # Old RSL-RL compatibility field. Both policies use global std.
        policy_cfg.pop("state_dependent_std", None)

        policy_class = policy_classes[policy_class_name]

        policy = policy_class(
            obs=obs,
            obs_groups=self.cfg["obs_groups"],
            num_actions=self.env.num_actions,
            **policy_cfg,
        ).to(self.device)

        teacher_env_ratio = float(
            algorithm_cfg["teacher_env_ratio"]
        )
        num_teacher_envs = max(int(self.env.num_envs * teacher_env_ratio), 1)
        num_student_envs = self.env.num_envs - num_teacher_envs

        if num_student_envs < 1:
            raise ValueError(
            "CTS requires at least one Student environment."
            )

        student_env_ids = (
            torch.arange(
                num_student_envs,
                device=self.device
            ) * self.env.num_envs
            // num_student_envs
        )

        student_mask = torch.zeros(
            self.env.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        student_mask[student_env_ids] = True

        teacher_env_ids = torch.nonzero(
            ~student_mask,
            as_tuple=False,
        ).squeeze(-1)

        storage = RolloutStorageCTS(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.cfg["num_steps_per_env"],
            obs=obs,
            obs_groups=self.cfg["obs_groups"],
            actions_shape=[self.env.num_actions],
            history_length=policy.history_length,
            teacher_env_ids=teacher_env_ids,
            device=self.device,
        )

        # Inherited PPO fields that CTS/MoENGCTS do not implement.
        algorithm_cfg.pop("rnd_cfg", None)
        algorithm_cfg.pop("symmetry_cfg", None)
        algorithm_cfg.pop("share_cnn_encoders", None)

        algorithm_class = algorithm_classes[algorithm_class_name]

        return algorithm_class(
           policy=policy,
           storage=storage,
           device=self.device,
           multi_gpu_cfg=self.multi_gpu_cfg,
           **algorithm_cfg,
        )
