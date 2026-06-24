from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from prettytable import PrettyTable

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerBase, ManagerTermBase
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .animation_manager_cfg import AnimationTermCfg
from .motiondata_manager import MotionDataTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class AnimationTerm(ManagerTermBase):
    """Animation term that initializes robot states from AMP reference motion.

    At reset, samples a frame from the reference motion dataset and applies
    it to the robot's root state and joint state.  Can also visualize the
    reference pose as a ghost robot overlay.

    Direct mapping from go2_amp.py:
        - reset()          → reset_idx() 中的 AMP 分支
        - _apply_to_sim()  → _reset_dofs_amp() + _reset_root_states_amp()
    """

    cfg: AnimationTermCfg
    _env: ManagerBasedEnv

    def __init__(self, cfg: AnimationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env) # type:ignore

        # -- 步数索引 --
        if cfg.num_steps_to_use > 0:
            self.step_indices = torch.arange(
                0, cfg.num_steps_to_use, dtype=torch.long, device=env.device
            )
        elif cfg.num_steps_to_use < 0:
            self.step_indices = torch.arange(
                cfg.num_steps_to_use + 1, 1, dtype=torch.long, device=env.device
            )
        else:
            raise ValueError("num_steps_to_use cannot be zero.")

        # -- 绑定 MotionDataTerm --
        if not hasattr(env, "motion_data_manager"):
            raise AttributeError(
                "AnimationTerm requires env.motion_data_manager to be created "
                "before env.animation_manager."
            )
        self.motion_data_term: MotionDataTerm = env.motion_data_manager.get_term(
            cfg.motion_data_term
        )

        # -- 为每个 component 创建 buffer --
        self.num_steps = len(self.step_indices)
        self._create_buffers(env)

        # -- 运动 ID 和采样时间 --
        self.motion_ids = torch.zeros(
            self.num_envs, device=env.device, dtype=torch.long
        )
        self.motion_fetch_time = torch.zeros(
            (self.num_envs, self.num_steps), device=env.device, dtype=torch.float32
        )
        self.motion_durations = torch.zeros(
            self.num_envs, device=env.device, dtype=torch.float32
        )

        # Fill buffers with the first frame of motion 0.  Do not write to sim
        # from __init__; reset-time logic decides when reference states apply.
        self._fetch_motion_data()

        # -- 可视化 --
        if self.cfg.enable_visualization:
            self.vis_root_offset = torch.tensor(
                self.cfg.vis_root_offset, device=env.device, dtype=torch.float32
            ).unsqueeze(0)

            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/KeyBodyVisualizerFromTerm",
                markers={
                    "red_sphere": sim_utils.SphereCfg(
                        radius=0.03,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.0, 0.0)
                        ),
                    ),
                },
            )
            self.key_body_marker: VisualizationMarkers = VisualizationMarkers(marker_cfg)

    # ------------------------------------------------------------------
    # 缓冲区创建
    # ------------------------------------------------------------------

    def _create_buffers(self, env: ManagerBasedEnv) -> None:
        """为 cfg.motion_data_components 中每个分量创建 buffer."""
        buffer_shape_map = {
            "root_pos_w":      (3,),
            "root_vel_w":      (3,),
            "root_vel_b":      (3,),
            "root_ang_vel_w":  (3,),
            "root_ang_vel_b":  (3,),
            "root_quat":       (4,),
            "dof_pos":         (self.motion_data_term.num_dofs,),
            "dof_vel":         (self.motion_data_term.num_dofs,),
            "key_body_pos_b":  (self.motion_data_term.num_key_bodies, 3),
        }

        for component in self.cfg.motion_data_components:
            shape = buffer_shape_map.get(component)
            if shape is None:
                raise ValueError(f"Unknown motion data component: {component}")
            buffer = torch.zeros(
                (self.num_envs, self.num_steps, *shape),
                device=env.device,
                dtype=torch.float32,
            )
            setattr(self, f"{component}_buffer", buffer)

    # ------------------------------------------------------------------
    # reset / update
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """采样新的运动帧并写入 sim.

        对应 go2_amp.py reset_idx() 中的 AMP 分支:
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)
        """
        if env_ids is None or len(env_ids) == 0:
            return
        env_ids = self._to_env_ids_tensor(env_ids)

        # -- 采样运动 ID --
        self.motion_ids[env_ids] = self.motion_data_term.sample_motions(len(env_ids))
        self.motion_durations[env_ids] = self.motion_data_term.get_motion_durations(
            self.motion_ids[env_ids]
        )

        # -- 采样起始时间 --
        truncate_time = (self.num_steps - 1) * self._env.step_dt
        if self.cfg.random_initialize:
            if self.cfg.num_steps_to_use > 0:
                anchor_time = self.motion_data_term.sample_times(
                    self.motion_ids[env_ids], truncate_time_end=truncate_time
                )
            else:
                anchor_time = self.motion_data_term.sample_times(
                    self.motion_ids[env_ids], truncate_time_start=truncate_time
                )
        else:
            anchor_time = torch.zeros(len(env_ids), device=self._env.device)

        self.motion_fetch_time[env_ids, :] = (
            anchor_time.unsqueeze(-1)
            + self.step_indices.float().unsqueeze(0) * self._env.step_dt
        )

        # -- 拉取数据并写入 sim --
        self._fetch_motion_data(env_ids)
        self._apply_to_sim(env_ids)

    def update(self, dt: float) -> None:
        """每步调用 (仅在 random_fetch 或 enable_visualization 时有效).

        go2_amp 中不每步更新, 这里保留最小实现以兼容框架.
        """
        if self.cfg.random_fetch:
            truncate_time = (self.num_steps - 1) * dt
            if self.cfg.num_steps_to_use > 0:
                anchor_time = self.motion_data_term.sample_times(
                    self.motion_ids, truncate_time_end=truncate_time
                )
            else:
                anchor_time = self.motion_data_term.sample_times(
                    self.motion_ids, truncate_time_start=truncate_time
                )

            self.motion_fetch_time[:, :] = (
                anchor_time.unsqueeze(-1)
                + self.step_indices.float().unsqueeze(0) * dt
            )

            self._fetch_motion_data()

        if not self.cfg.random_fetch:
            self.motion_fetch_time += dt

        if self.cfg.enable_visualization:
            self._visualize()

    # ------------------------------------------------------------------
    # 数据获取
    # ------------------------------------------------------------------

    def _fetch_motion_data(self, env_ids: Sequence[int] | None = None) -> None:
        """从 MotionDataTerm 批量拉取参考运动帧, 填入各 component buffer."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._env.device)
        else:
            env_ids = self._to_env_ids_tensor(env_ids)

        motion_times_flat = self.motion_fetch_time[env_ids].reshape(-1)
        motion_ids_flat = self.motion_ids[env_ids].repeat_interleave(self.num_steps)

        motion_data_dict = self.motion_data_term.get_motion_state(
            motion_ids=motion_ids_flat,
            motion_times=motion_times_flat,
        )

        for component in self.cfg.motion_data_components:
            if component in motion_data_dict:
                buffer_name = f"{component}_buffer"
                data = motion_data_dict[component]
                if component == "key_body_pos_b":
                    data = data.view(-1, self.motion_data_term.num_key_bodies, 3)
                data_reshaped = data.view(
                    len(env_ids), self.num_steps, *data.shape[1:]
                )
                getattr(self, buffer_name)[env_ids, :] = data_reshaped

    # ------------------------------------------------------------------
    # 写入 sim (核心: 对应 _reset_dofs_amp + _reset_root_states_amp)
    # ------------------------------------------------------------------

    def _apply_to_sim(self, env_ids: Sequence[int]) -> None:
        """将第 0 帧的运动数据写入仿真.

        对应 go2_amp.py:
            _reset_dofs_amp(env_ids, frames)
            _reset_root_states_amp(env_ids, frames)
        """
        robot: Articulation = self._env.scene["robot"]

        # -- 基座位姿 --
        if hasattr(self, "root_pos_w_buffer") and hasattr(self, "root_quat_buffer"):
            root_pos = self.root_pos_w_buffer[env_ids, 0, :]
            root_rot = self.root_quat_buffer[env_ids, 0, :]
            default_root = robot.data.default_root_state[env_ids].clone()
            default_root[:, :3] = root_pos + self._env.scene.env_origins[env_ids, :3]
            default_root[:, 3:7] = root_rot

            # 线速度和角速度 (如果配置了)
            if hasattr(self, "root_vel_w_buffer"):
                vel_w = self.root_vel_w_buffer[env_ids, 0, :]
                default_root[:, 7:10] = vel_w
            if hasattr(self, "root_ang_vel_w_buffer"):
                ang_vel_w = self.root_ang_vel_w_buffer[env_ids, 0, :]
                default_root[:, 10:13] = ang_vel_w

            robot.write_root_state_to_sim(default_root, env_ids=env_ids)

        # -- 关节状态 --
        if hasattr(self, "dof_pos_buffer"):
            joint_pos = self.dof_pos_buffer[env_ids, 0, :]
            joint_vel = torch.zeros_like(joint_pos)
            if hasattr(self, "dof_vel_buffer"):
                joint_vel = self.dof_vel_buffer[env_ids, 0, :]
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------

    def _visualize(self) -> None:
        """在 ghost robot 上显示参考姿态."""
        if "robot_anim" not in self._env.scene.keys():
            return
        robot_anim: Articulation = self._env.scene["robot_anim"]

        required = ["root_pos_w_buffer", "root_quat_buffer", "dof_pos_buffer"]
        if not all(hasattr(self, b) for b in required):
            return

        root_pos_w = self.root_pos_w_buffer[:, 0, :]
        root_quat = self.root_quat_buffer[:, 0, :]
        dof_pos = self.dof_pos_buffer[:, 0, :]

        root_states = robot_anim.data.default_root_state.clone()
        root_states[:, :3] = (
            root_pos_w + self._env.scene.env_origins[:, :3] + self.vis_root_offset
        )
        root_states[:, 3:7] = root_quat
        root_states[:, 7:13] = 0.0
        robot_anim.write_root_state_to_sim(root_states)

        joint_pos = robot_anim.data.default_joint_pos.clone()
        joint_pos[:, :] = dof_pos
        robot_anim.write_joint_state_to_sim(
            joint_pos, torch.zeros_like(joint_pos)
        )

        if hasattr(self, "key_body_pos_b_buffer"):
            key_body_pos_b = self.key_body_pos_b_buffer[:, 0, :, :]
            num_key = key_body_pos_b.shape[1]
            key_body_pos_w = root_states[:, :3].unsqueeze(1) + math_utils.quat_apply(
                root_quat.unsqueeze(1).expand(-1, num_key, -1).reshape(-1, 4),
                key_body_pos_b.reshape(-1, 3),
            ).view(self.num_envs, num_key, 3)
            self.key_body_marker.visualize(translations=key_body_pos_w.reshape(-1, 3))

    # ------------------------------------------------------------------
    # getter (供 EventManager term 函数使用)
    # ------------------------------------------------------------------

    def get_root_pos_w(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        b = getattr(self, "root_pos_w_buffer")
        return b if env_ids is None else b[env_ids]

    def get_root_quat(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        b = getattr(self, "root_quat_buffer")
        return b if env_ids is None else b[env_ids]

    def get_dof_pos(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        b = getattr(self, "dof_pos_buffer")
        return b if env_ids is None else b[env_ids]

    def get_dof_vel(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        b = getattr(self, "dof_vel_buffer")
        return b if env_ids is None else b[env_ids]

    def get_root_vel_w(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        b = getattr(self, "root_vel_w_buffer")
        return b if env_ids is None else b[env_ids]

    def get_root_ang_vel_w(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        b = getattr(self, "root_ang_vel_w_buffer")
        return b if env_ids is None else b[env_ids]

    def get_key_body_pos_b(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        b = getattr(self, "key_body_pos_b_buffer")
        return b if env_ids is None else b[env_ids]

    def _to_env_ids_tensor(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self._env.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self._env.device, dtype=torch.long)


class AnimationManager(ManagerBase):
    """Manager for animation terms.

    管理所有 AnimationTerm, 在 reset 和 step 时调度.

    对应 go2_amp.py 中:
        env.amp_loader + reset_idx() 中的 _reset_dofs_amp / _reset_root_states_amp
        但现在被拆成:
            - MotionDataManager: 管理参考运动数据的加载和采样
            - AnimationManager:   管理如何在 reset 时将数据写入 sim
    """

    _env: ManagerBasedEnv
    _terms: dict[str, AnimationTerm]
    _term_cfgs: dict[str, AnimationTermCfg]

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        if cfg is None:
            raise ValueError("AnimationManager configuration is required.")
        self._terms = {}
        self._term_cfgs = {}
        super().__init__(cfg, env)

    def __str__(self) -> str:
        msg = f"<AnimationManager> contains {len(self._terms)} active terms.\n"

        table = PrettyTable()
        table.title = "Animation Manager Terms"
        table.field_names = [
            "Index",
            "Term Name",
            "Motion Data Term",
            "Num Steps",
            "Random Init",
            "Random Fetch",
            "Enabled",
            "Probability",
        ]
        for idx, (name, term_cfg) in enumerate(self._term_cfgs.items()):
            table.add_row([
                idx,
                name,
                term_cfg.motion_data_term,
                term_cfg.num_steps_to_use,
                term_cfg.random_initialize,
                term_cfg.random_fetch,
                term_cfg.enable,
                term_cfg.probability,
            ])
        msg += table.get_string() + "\n"
        return msg

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """重置时采样新运动帧并写入 sim.

        对应 go2_amp.py reset_idx() 中:
            if prob > rand: → _reset_dofs_amp + _reset_root_states_amp
            else:           → _reset_dofs + _reset_root_states
        """
        if env_ids is None or len(env_ids) == 0:
            return {}
        env_ids = self._to_env_ids_tensor(env_ids)

        for term in self._terms.values():
            if not term.cfg.enable:
                continue
            if term.cfg.probability <= 0.0:
                continue
            if term.cfg.probability >= 1.0:
                selected_env_ids = env_ids
            else:
                mask = torch.rand(len(env_ids), device=self._env.device) < term.cfg.probability
                selected_env_ids = env_ids[mask]
            if len(selected_env_ids) > 0:
                term.reset(selected_env_ids)

        return {}

    def update(self, dt: float) -> None:
        """每步更新 (仅 visualization / random_fetch 场景使用).

        go2_amp 中不需要每步调用, 但保留接口以兼容框架.
        """
        for term in self._terms.values():
            if not term.cfg.enable:
                continue
            term.update(dt)

    # ------------------------------------------------------------------
    # 访问
    # ------------------------------------------------------------------

    @property
    def active_terms(self) -> list[str]:
        return list(self._terms.keys())

    def get_term(self, term_name: str) -> AnimationTerm:
        if term_name not in self._terms:
            raise KeyError(f"Animation term '{term_name}' not found. "
                           f"Available: {list(self._terms.keys())}")
        return self._terms[term_name]

    def _to_env_ids_tensor(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self._env.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self._env.device, dtype=torch.long)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _prepare_terms(self) -> None:
        cfg_items = (
            self.cfg.items() if isinstance(self.cfg, dict)
            else self.cfg.__dict__.items()
        )
        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, AnimationTermCfg):
                raise TypeError(
                    f"Term '{term_name}' is not AnimationTermCfg. "
                    f"Got: {type(term_cfg)}"
                )
            self._terms[term_name] = AnimationTerm(term_cfg, self._env)
            self._term_cfgs[term_name] = term_cfg
