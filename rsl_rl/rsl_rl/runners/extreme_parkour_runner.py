from __future__ import annotations

import os
import time
from typing import Callable

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import ExtremeParkourPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ExtremeParkourActorCritic
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.storage import RolloutStorage

class ExtremeParkourRunner(OnPolicyRunner):
    CHECKPOINT_FORMAT_VERSION = 1

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.observation_contract_version = train_cfg[
            "observation_contract_version"
        ]

        super().__init__(
            env=env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=device,
        )

    def _construct_algorithm(
        self,
        obs: TensorDict,
    ) -> ExtremeParkourPPO:
        policy_cfg = self.policy_cfg.copy()
        policy_class_name = policy_cfg.pop("class_name")

        if policy_class_name != "ExtremeParkourActorCritic":
            raise ValueError(
                "ExtremeParkourRunner requires "
                "'ExtremeParkourActorCritic', but received "
                f"'{policy_class_name}'."
            )

        policy = ExtremeParkourActorCritic(
            obs=obs,
            obs_groups=self.cfg["obs_groups"],
            num_actions=self.env.num_actions,
            **policy_cfg,
        ).to(self.device)

        storage = RolloutStorage(
            training_type="rl",
            num_envs=self.env.num_envs,
            num_transitions_per_env=(
                self.cfg["num_steps_per_env"]
            ),
            obs=obs,
            actions_shape=[self.env.num_actions],
            device=self.device,
        )

        algorithm_cfg = self.alg_cfg.copy()
        algorithm_class_name = algorithm_cfg.pop("class_name")

        if algorithm_class_name != "ExtremeParkourPPO":
            raise ValueError(
                "ExtremeParkourRunner requires "
                "'ExtremeParkourPPO', but received "
                f"'{algorithm_class_name}'."
            )

        algorithm = ExtremeParkourPPO(
            policy=policy,
            storage=storage,
            device=self.device,
            multi_gpu_cfg=self.multi_gpu_cfg,
            **algorithm_cfg,
        )

        self.observation_contract =  self._build_observation_contract(obs)

        return algorithm

    def _build_observation_contract(self, obs: TensorDict,) -> dict:
        """保存能够识别观测布局的完整契约。"""

        observation_shapes = {
            key: list(value.shape[1:])
            for key, value in obs.items()
        }

        return {
            "version": self.observation_contract_version,
            "observation_shapes": observation_shapes,
            "obs_groups": self.cfg["obs_groups"],
            "num_actions": self.env.num_actions,
        }

    def _validate_checkpoint_contract(
        self,
        checkpoint: dict,
    ) -> None:
        saved_contract = checkpoint.get(
            "observation_contract"
        )

        if saved_contract is None:
            raise RuntimeError(
                "Checkpoint does not contain an observation "
                "contract. Refusing to load it into the "
                "Extreme Parkour task."
            )

        if saved_contract != self.observation_contract:
            raise RuntimeError(
                "Checkpoint observation contract mismatch.\n"
                f"Saved contract: {saved_contract}\n"
                f"Current contract: {self.observation_contract}"
            )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Initialize logging (tensorboard + console output)
        self.logger.init_logging_writer()

        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iteration = self.current_learning_iteration
        end_iteration = start_iteration + num_learning_iterations
        for iteration in range(start_iteration, end_iteration):
            collection_start = time.time()

            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                   # obs_t -> action_t
                   actions = self.alg.act(obs)

                   # action_t -> obs_{t+1}, reward_t, done_t
                   next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                   next_obs = next_obs.to(self.device)
                   rewards = rewards.to(self.device)
                   dones = dones.to(self.device)
                   # 这里的obs是当前观测
                   self.alg.process_env_step(obs, rewards, dones, extras)

                   self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards = None)

                   obs = next_obs

                collection_time = time.time() - collection_start

                learning_start = time.time()

                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            learning_time = time.time() - learning_start

            completed_iteration = iteration + 1

            self.logger.log(
                it=iteration,
                start_it=start_iteration,
                total_it=end_iteration,
                collect_time=collection_time,
                learn_time=learning_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.policy.action_std,
                rnd_weight=None,
            )

            if (completed_iteration % self.cfg["save_interval"] == 0):
                self.save(os.path.join(self.logger.log_dir, f"model_{completed_iteration}.pt")) # type: ignore

        # Save the final model after training
        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(
        self,
        path: str,
        infos: dict | None = None,
    ) -> None:
        """保存完整可续训检查点。"""

        checkpoint = {
            "checkpoint_format_version": self.CHECKPOINT_FORMAT_VERSION,
            "observation_contract": self.observation_contract,
            # 网络
            "model_state_dict": self.alg.policy.state_dict(),
            "estimator_state_dict": self.alg.estimator.state_dict(), # type:ignore

            # 三个优化器
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_optimizer_state_dict": self.alg.estimator_optimizer.state_dict(), # type:ignore
            "history_optimizer_state_dict": self.alg.history_optimizer.state_dict(), # type:ignore

            # 训练进度
            "iter": self.current_learning_iteration,
            "algorithm_update_counter": self.alg.update_counter, # type:ignore
            "infos": infos,
        }

        self.logger.save_model(path, self.current_learning_iteration)
        torch.save(checkpoint, path)

    def load(self, path: str, load_optimizer: bool = False, map_location: str | None = None) -> dict | None:
        """恢复模型，以及可选的完整训练状态。"""

        checkpoint = torch.load(
            path,
            weights_only=False,
            map_location=map_location if map_location is not None else self.device,
        )

        checkpoint_format_version = checkpoint.get("checkpoint_format_version")

        if (checkpoint_format_version != self.CHECKPOINT_FORMAT_VERSION):
            raise RuntimeError(
                "Unsupported Extreme Parkour checkpoint "
                f"format: {checkpoint_format_version}. "
                f"Expected "
                f"{self.CHECKPOINT_FORMAT_VERSION}."
            )

        self._validate_checkpoint_contract(checkpoint)

        self.alg.policy.load_state_dict(checkpoint["model_state_dict"])

        self.alg.estimator.load_state_dict(checkpoint["estimator_state_dict"]) # type:ignore

        if load_optimizer:
            required_optimizer_keys = (
                "optimizer_state_dict",
                "estimator_optimizer_state_dict",
                "history_optimizer_state_dict",
            )

            missing_keys = [key for key in required_optimizer_keys if key not in checkpoint]

            if missing_keys:
                raise RuntimeError(
                    "Checkpoint cannot resume training "
                    "because optimizer states are missing: "
                    f"{missing_keys}"
                )

            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            self.alg.estimator_optimizer.load_state_dict(checkpoint["estimator_optimizer_state_dict"]) # type:ignore

            self.alg.history_optimizer.load_state_dict(checkpoint["history_optimizer_state_dict"]) # type:ignore

        self.current_learning_iteration = checkpoint["iter"]

        self.alg.update_counter = checkpoint["algorithm_update_counter"] # type:ignore

        return checkpoint.get("infos")

    def train_mode(self) -> None:
        self.alg.policy.train()
        self.alg.estimator.train() # type:ignore

    def eval_mode(self) -> None:
        self.alg.policy.eval()
        self.alg.estimator.eval() # type:ignore

    def get_inference_policy(self, device: str | None = None,) -> Callable[[TensorDict], torch.Tensor]:
        """返回只需要TensorDict观测的部署式教师策略。"""

        self.eval_mode()

        if device is not None:
            self.alg.policy.to(device)
            self.alg.estimator.to(device) # type:ignore

        def inference_policy(obs: TensorDict) -> torch.Tensor:
            with torch.inference_mode():
                explicit_estimate = self.alg.estimator(obs[self.alg.policy.estimator_input_key]) # type:ignore

                return self.alg.policy.act_inference(
                    obs,
                    # play时使用历史latent，
                    # 不依赖priv_latent编码结果。
                    use_history=True, # type:ignore
                    explicit_override=explicit_estimate, # type:ignore
                )

        return inference_policy
