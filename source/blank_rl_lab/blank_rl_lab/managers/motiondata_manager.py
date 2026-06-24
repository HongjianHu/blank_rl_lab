# ============ motion_data_term.py ============

from __future__ import annotations

import json
import os
from enum import IntEnum
from typing import TYPE_CHECKING

from prettytable import PrettyTable

import torch
import numpy as np

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerBase, ManagerTermBase

from .motiondata_manager_cfg import MotionDataTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ============================================================
# 辅助: quaternion / interpolation
# ============================================================

@torch.jit.script
def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two quaternion batches.

    q0, q1: (N, 4) in (w, x, y, z)
    blend:   (N, 1) or (N,)
    """
    if blend.dim() == 1:
        blend = blend.unsqueeze(-1)

    # 确保走最短弧
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    neg_mask = dot < 0.0
    q1 = torch.where(neg_mask, -q1, q1)
    dot = torch.where(neg_mask, -dot, dot)

    # 夹角过小退化为线性
    dot = torch.clamp(dot, -1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    small_angle = sin_theta < 1e-6
    safe_sin_theta = torch.where(small_angle, torch.ones_like(sin_theta), sin_theta)
    t0 = torch.where(small_angle, 1.0 - blend, torch.sin((1.0 - blend) * theta) / safe_sin_theta)
    t1 = torch.where(small_angle, blend, torch.sin(blend * theta) / safe_sin_theta)

    out = t0 * q0 + t1 * q1
    return out / torch.norm(out, dim=-1, keepdim=True).clamp(min=1e-12)


class LoopMode(IntEnum):
    """AMP motion loop mode, 对应 JSON 中 "LoopMode" 字段."""
    CLAMP = 0  # 到头停止
    WRAP = 1   # 到头循环


@torch.jit.script
def _calc_phase(
    times: torch.Tensor,
    motion_durations: torch.Tensor,
    loop_modes: torch.Tensor,
) -> torch.Tensor:
    """计算给定时间在运动中的相位 [0, 1].

    对应 AMPLoader 中的时间→帧索引计算.
    """
    phase = times / motion_durations

    wrap_mask = loop_modes == 1
    p_wrap = phase[wrap_mask]
    p_wrap = p_wrap - torch.floor(p_wrap)
    phase[wrap_mask] = p_wrap

    return torch.clamp(phase, 0.0, 1.0)


# ============================================================
# MotionDataTerm
# ============================================================

class MotionDataTerm(ManagerTermBase):
    """Loads AMP-format motion data and provides sampling / interpolation.

    Comparison with AMPLoader (motion_loader.py):
        _load_motion_data()          → AMPLoader.__init__()
        sample_motions()             → weighted_traj_idx_sample_batch()
        sample_times()               → traj_time_sample_batch()
        get_motion_state()           → get_full_frame_at_time_batch()
        61-dim frame parsing         → AMPLoader static getter methods

    AMP 61-dim frame layout (one frame):
        [0:3]   root_pos_w      base world position
        [3:7]   root_quat       base rotation (x, y, z, w) → stored as (w, x, y, z)
        [7:19]  dof_pos         12 joint angles
        [19:31] key_body_pos_b  4 foot positions in base frame
        [31:34] lin_vel_b       base linear velocity in base frame
        [34:37] ang_vel_b       base angular velocity in base frame
        [37:49] dof_vel         12 joint velocities
        [49:61] foot_vel_b      4 foot velocities in base frame (unused)
    """

    cfg: MotionDataTermCfg
    _env: ManagerBasedEnv

    # ── 61-dim frame component slices (consistent with AMPLoader) ──
    _ROOT_POS_IDX      = slice(0, 3)
    _ROOT_QUAT_IDX     = slice(3, 7)
    _DOF_POS_IDX       = slice(7, 19)
    _KEY_BODY_POS_IDX  = slice(19, 31)
    _LIN_VEL_IDX       = slice(31, 34)
    _ANG_VEL_IDX       = slice(34, 37)
    _DOF_VEL_IDX       = slice(37, 49)
    # [49:61] foot_vel — not used

    _FRAME_DIM = 61

    def __init__(self, cfg: MotionDataTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)                              # type:ignore

        if not os.path.exists(cfg.motion_data_dir):
            raise FileNotFoundError(
                f"Motion data directory '{cfg.motion_data_dir}' does not exist."
            )

        self.time_between_frames = cfg.time_between_frames
        self._load_motion_data()

    # ------------------------------------------------------------------
    # Data loading (AMPLoader.__init__)
    # ------------------------------------------------------------------

    def _load_motion_data(self) -> None:
        """Load every .txt AMP motion file under the configured directory.

        Comparison with AMPLoader.__init__():
            for motion_file in motion_files:
                motion_json = json.load(f)
                motion_data  = np.array(motion_json["Frames"])
                ... normalize / standardize quaternion ...
                self.trajectories_full.append(...)
                self.trajectory_weights.append(motion_json["MotionWeight"])
                self.trajectory_frame_durations.append(motion_json["FrameDuration"])
        """
        # Discover .txt files.
        all_files = sorted(
            f for f in os.listdir(self.cfg.motion_data_dir)
            if f.endswith(".txt")
        )
        if not all_files:
            raise ValueError(
                f"No .txt motion files found in '{self.cfg.motion_data_dir}'."
            )

        # If weights are explicitly configured, only load those files.
        weight_dict = self.cfg.motion_data_weights
        if weight_dict:
            file_names = [f"{name}.txt" for name in weight_dict]
            missing = set(file_names) - set(all_files)
            if missing:
                raise ValueError(
                    f"Motion files not found: {missing}. "
                    f"Available: {all_files}"
                )
            target_files = file_names
        else:
            target_files = all_files
            weight_dict = {}

        # ── Per-file accumulators ──
        (
            root_pos_w_list, root_quat_list, dof_pos_list,
            key_body_pos_b_list,
            root_vel_b_list, root_ang_vel_b_list,
            root_vel_w_list, root_ang_vel_w_list,
            dof_vel_list,
        ) = ([], [], [], [], [], [], [], [], [])

        durations_list   = []
        dt_list          = []
        num_frames_list  = []
        weights_list     = []
        loop_modes_list  = []

        for fname in target_files:
            fpath = os.path.join(self.cfg.motion_data_dir, fname)
            with open(fpath, "r") as f:
                motion_json = json.load(f)

            frames = np.array(motion_json["Frames"], dtype=np.float32)
            if frames.shape[1] != self._FRAME_DIM:
                raise ValueError(
                    f"Expected {self._FRAME_DIM}-dim frames in '{fname}', "
                    f"got {frames.shape[1]}."
                )

            num_frames = frames.shape[0]
            frame_duration = float(motion_json["FrameDuration"])

            # ── Extract components ──
            root_pos_w = torch.from_numpy(
                frames[:, self._ROOT_POS_IDX]
            ).to(self.device)

            # Quaternion: AMP stores (x, y, z, w) → convert to (w, x, y, z)
            # and normalise + standardise.
            root_quat_xyzw = torch.from_numpy(
                frames[:, self._ROOT_QUAT_IDX]
            ).to(self.device)
            root_quat = self._normalize_and_standardize_quat(root_quat_xyzw)

            dof_pos = torch.from_numpy(
                frames[:, self._DOF_POS_IDX]
            ).to(self.device)
            key_body_pos_b = torch.from_numpy(
                frames[:, self._KEY_BODY_POS_IDX]
            ).to(self.device)

            # Velocities in base frame (as stored in AMP data).
            lin_vel_b = torch.from_numpy(
                frames[:, self._LIN_VEL_IDX]
            ).to(self.device)
            ang_vel_b = torch.from_numpy(
                frames[:, self._ANG_VEL_IDX]
            ).to(self.device)
            dof_vel = torch.from_numpy(
                frames[:, self._DOF_VEL_IDX]
            ).to(self.device)

            # Convert base-frame velocities to world frame.
            root_vel_w     = math_utils.quat_apply(root_quat, lin_vel_b)
            root_ang_vel_w = math_utils.quat_apply(root_quat, ang_vel_b)

            # Freeze all tensors.
            for t in (
                root_pos_w, root_quat, dof_pos, key_body_pos_b,
                lin_vel_b, ang_vel_b, dof_vel, root_vel_w, root_ang_vel_w,
            ):
                t.requires_grad_(False)

            root_pos_w_list.append(root_pos_w)
            root_quat_list.append(root_quat)
            dof_pos_list.append(dof_pos)
            key_body_pos_b_list.append(key_body_pos_b)
            root_vel_b_list.append(lin_vel_b)
            root_ang_vel_b_list.append(ang_vel_b)
            root_vel_w_list.append(root_vel_w)          # ← was missing
            root_ang_vel_w_list.append(root_ang_vel_w)  # ← was missing
            dof_vel_list.append(dof_vel)

            # ── Metadata ──
            dt_list.append(frame_duration)
            num_frames_list.append(num_frames)
            # Total duration = frame_duration × (num_frames − 1).
            durations_list.append(frame_duration * (num_frames - 1))

            # Weight: explicit config wins, otherwise use file's MotionWeight.
            motion_name = fname.replace(".txt", "")
            if motion_name in weight_dict:
                weights_list.append(float(weight_dict[motion_name]))
            else:
                weights_list.append(float(motion_json.get("MotionWeight", 1.0)))

            # Loop mode.
            loop_str = motion_json.get("LoopMode", "Wrap")
            loop_modes_list.append(
                LoopMode.WRAP if loop_str == "Wrap" else LoopMode.CLAMP
            )

            print(
                f"[MotionDataTerm] Loaded '{fname}': "
                f"{num_frames} frames, "
                f"{(num_frames - 1) * frame_duration:.1f}s, "
                f"weight={weights_list[-1]:.2f}"
            )

        # ── Concatenate all motions into contiguous arrays ──
        self.root_pos_w      = torch.cat(root_pos_w_list,      dim=0)
        self.root_quat       = torch.cat(root_quat_list,       dim=0)
        self.dof_pos         = torch.cat(dof_pos_list,         dim=0)
        self.dof_vel         = torch.cat(dof_vel_list,         dim=0)
        self.key_body_pos_b  = torch.cat(key_body_pos_b_list,  dim=0)
        self._root_vel_b     = torch.cat(root_vel_b_list,      dim=0)  # base frame
        self._root_ang_vel_b = torch.cat(root_ang_vel_b_list,  dim=0)  # base frame
        self.root_vel_w      = torch.cat(root_vel_w_list,      dim=0)  # world frame
        self.root_ang_vel_w  = torch.cat(root_ang_vel_w_list,  dim=0)  # world frame

        # Metadata tensors.
        self.motion_dt          = torch.tensor(dt_list,          device=self.device, dtype=torch.float32)
        self.motion_durations   = torch.tensor(durations_list,   device=self.device, dtype=torch.float32)
        self.motion_num_frames  = torch.tensor(num_frames_list,  device=self.device, dtype=torch.int64)
        self.motion_loop_modes  = torch.tensor(loop_modes_list,  device=self.device, dtype=torch.int64)
        self.motion_weights     = torch.tensor(weights_list,     device=self.device, dtype=torch.float32)
        self.motion_weights = self.motion_weights / self.motion_weights.sum()

        # Start frame index of each motion inside the concatenated arrays.
        shifted = self.motion_num_frames.roll(1)
        shifted[0] = 0
        self.motion_start_indices = torch.cumsum(shifted, dim=0)

        # Derived dimensions.
        self.num_dofs       = self.dof_pos.shape[1]             # 12
        self.num_key_bodies = self.key_body_pos_b.shape[1] // 3  # 4 feet
        self.observation_dim = (
            self.num_dofs
            + self.num_key_bodies * 3
            + 3
            + 3
            + self.num_dofs
            + 1
        )

        print(
            f"[MotionDataTerm] Total: {self.get_num_motions()} motions, "
            f"{self.root_pos_w.shape[0]} frames, "
            f"{self.get_total_duration():.1f}s, "
            f"{self.num_dofs} dofs, "
            f"{self.num_key_bodies} key bodies."
        )

    @staticmethod
    def _normalize_and_standardize_quat(q_xyzw: torch.Tensor) -> torch.Tensor:
        """Normalise and standardise a quaternion, then convert to (w, x, y, z).

        Comparison with AMPLoader:
            root_rot = pose3d.QuaternionNormalize(root_rot)
            root_rot = motion_util.standardize_quaternion(root_rot)

        Input:  (N, 4) in (x, y, z, w) — AMP format.
        Output: (N, 4) in (w, x, y, z) — Isaac Lab / standard format.
        """
        # Normalise to unit length.
        q = q_xyzw / torch.norm(q_xyzw, dim=-1, keepdim=True).clamp(min=1e-12)
        # Standardise: ensure scalar component w (index 3) is non-negative.
        w_neg: torch.Tensor = q[:, 3] < 0
        q[w_neg] = -q[w_neg]
        # Reorder from (x, y, z, w) to (w, x, y, z).
        return torch.stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]], dim=-1)

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_num_motions(self) -> int:
        """Number of loaded motion clips."""
        return int(self.motion_num_frames.shape[0])

    def get_total_duration(self) -> float:
        """Total duration of all motions combined (seconds)."""
        return float(self.motion_durations.sum().item())

    def get_motion_durations(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Return the duration of each requested motion.

        Args:
            motion_ids: (N,) int tensor of motion indices.

        Returns:
            (N,) float tensor of durations in seconds.
        """
        return self.motion_durations[motion_ids]

    # ------------------------------------------------------------------
    # Sampling (AMPLoader.weighted_traj_idx_sample_batch / traj_time_sample_batch)
    # ------------------------------------------------------------------

    def sample_motions(self, n: int) -> torch.Tensor:
        """Weighted random sampling of *n* motion IDs.

        Corresponds to: weighted_traj_idx_sample_batch(size).
        """
        return torch.multinomial(self.motion_weights, num_samples=n, replacement=True)

    def sample_times(
        self,
        motion_ids: torch.Tensor,
        truncate_time_start: float | None = None,
        truncate_time_end: float | None = None,
    ) -> torch.Tensor:
        """Randomly sample a time within the duration of each given motion.

        Corresponds to: traj_time_sample_batch(traj_idxs)
            subst = time_between_frames + trajectory_frame_durations[traj_idx]
            time_samples = trajectory_lens[traj_idx] * uniform - subst

        Args:
            motion_ids:         (N,) int tensor of motion indices.
            truncate_time_start: Seconds to trim from the start of each clip.
            truncate_time_end:   Seconds to trim from the end of each clip.

        Returns:
            (N,) float tensor of sampled times in seconds.
        """
        durations = self.motion_durations[motion_ids]
        dts       = self.motion_dt[motion_ids]

        time_start = torch.zeros_like(durations)
        time_end   = durations.clone()

        if truncate_time_start is not None:
            time_start = torch.clamp(time_start + truncate_time_start, min=0.0, max=durations) # type:ignore
        if truncate_time_end is not None:
            time_end = torch.clamp(time_end - truncate_time_end, min=0.0)

        # Reserve at least one frame-time so interpolation always has a valid pair.
        time_end = torch.clamp(time_end - dts, min=time_start)

        phase = torch.rand(motion_ids.shape, device=self.device)
        return time_start + phase * (time_end - time_start)

    def sample_transition_times(self, motion_ids: torch.Tensor, dt: float) -> torch.Tensor:
        """Sample transition start times exactly like AMPLoader.traj_time_sample_batch().

        Original formula:
            subst = time_between_frames + trajectory_frame_duration
            sample = max(0, trajectory_length * uniform - subst)
        """
        durations = self.motion_durations[motion_ids]
        dts = self.motion_dt[motion_ids]
        subst = dt + dts
        samples = durations * torch.rand(motion_ids.shape, device=self.device) - subst
        return torch.clamp(samples, min=0.0)

    # ------------------------------------------------------------------
    # Frame-index computation
    # ------------------------------------------------------------------

    def _calc_frame_blend(
        self, motion_ids: torch.Tensor, times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (frame_idx0, frame_idx1, blend) as global indices.

        Corresponds to (inside AMPLoader):
            p = times / trajectory_lens
            idx_low, idx_high = floor(p * n), ceil(p * n)
            blend = p * n - idx_low

        Args:
            motion_ids: (N,) int tensor of motion indices.
            times:      (N,) float tensor of times in seconds.

        Returns:
            frame_idx0: (N,) global index of the lower frame.
            frame_idx1: (N,) global index of the upper frame.
            blend:      (N,) interpolation factor [0, 1).
        """
        num_frames    = self.motion_num_frames[motion_ids]
        durations     = self.motion_durations[motion_ids]
        loop_modes    = self.motion_loop_modes[motion_ids]
        start_indices = self.motion_start_indices[motion_ids]

        phase = _calc_phase(times, durations, loop_modes)

        # Match AMPLoader exactly: idx = floor((time / traj_len) * num_frames).
        # This is slightly different from the physical (num_frames - 1) formula,
        # but keeps expert samples aligned with the original AMP implementation.
        frame_pos = phase * num_frames.float()
        local_idx0 = torch.floor(frame_pos).long()
        local_idx1 = torch.ceil(frame_pos).long()
        max_local_idx = num_frames - 1
        local_idx0 = torch.clamp(local_idx0, min=0)
        local_idx1 = torch.clamp(local_idx1, min=0)
        local_idx0 = torch.minimum(local_idx0, max_local_idx)
        local_idx1 = torch.minimum(local_idx1, max_local_idx)
        blend = frame_pos - local_idx0.float()

        # Convert to global indices.
        frame_idx0 = local_idx0 + start_indices
        frame_idx1 = local_idx1 + start_indices

        return frame_idx0, frame_idx1, blend

    # ------------------------------------------------------------------
    # Motion state query (AMPLoader.get_full_frame_at_time_batch)
    # ------------------------------------------------------------------

    def get_motion_state(
        self, motion_ids: torch.Tensor, motion_times: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return the interpolated motion state for each (motion_id, time).

        All components use linear interpolation except quaternions, which use slerp.

        Corresponds to AMPLoader.get_full_frame_at_time_batch():
            pos_blend = slerp(pos0, pos1, blend)
            rot_blend = quaternion_slerp(rot0, rot1, blend)
            amp_blend = slerp(amp0, amp1, blend)

        Args:
            motion_ids:  (N,) int tensor of motion indices.
            motion_times: (N,) float tensor of times in seconds.

        Returns:
            Dict with keys:
                root_pos_w, root_quat, root_vel_w, root_vel_b,
                root_ang_vel_w, root_ang_vel_b, dof_pos, dof_vel,
                key_body_pos_b
        """
        n = int(motion_ids.shape[0])
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)

        # ── Fetch the two bounding frames from concatenated arrays ──
        pos0, pos1 = self.root_pos_w[frame_idx0], self.root_pos_w[frame_idx1]
        quat0, quat1 = self.root_quat[frame_idx0], self.root_quat[frame_idx1]
        dof0, dof1 = self.dof_pos[frame_idx0], self.dof_pos[frame_idx1]
        dvel0, dvel1 = self.dof_vel[frame_idx0], self.dof_vel[frame_idx1]
        kbp0, kbp1 = self.key_body_pos_b[frame_idx0], self.key_body_pos_b[frame_idx1]
        rvw0, rvw1 = self.root_vel_w[frame_idx0], self.root_vel_w[frame_idx1]
        ravw0, ravw1 = self.root_ang_vel_w[frame_idx0], self.root_ang_vel_w[frame_idx1]
        rvb0, rvb1 = self._root_vel_b[frame_idx0], self._root_vel_b[frame_idx1]
        ravb0, ravb1 = self._root_ang_vel_b[frame_idx0], self._root_ang_vel_b[frame_idx1]

        # ── Interpolate ──
        blend_q = blend.unsqueeze(-1)  # (N, 1) for slerp
        blend_v = blend.unsqueeze(-1)  # (N, 1) for lerp

        root_quat      = quat_slerp(quat0, quat1, blend_q)
        root_pos_w     = torch.lerp(pos0, pos1, blend_v)
        dof_pos        = torch.lerp(dof0, dof1, blend_v)
        dof_vel        = torch.lerp(dvel0, dvel1, blend_v)
        root_vel_w     = torch.lerp(rvw0, rvw1, blend_v)
        root_ang_vel_w = torch.lerp(ravw0, ravw1, blend_v)
        root_vel_b     = torch.lerp(rvb0, rvb1, blend_v)
        root_ang_vel_b = torch.lerp(ravb0, ravb1, blend_v)

        # key_body_pos is stored flat (N, num_key_bodies * 3);
        # reshape to (N, num_key_bodies, 3) for interpolation, then flatten back.
        kbp0 = kbp0.view(n, self.num_key_bodies, 3)
        kbp1 = kbp1.view(n, self.num_key_bodies, 3)
        key_body_pos_b = torch.lerp(kbp0, kbp1, blend_v.unsqueeze(1))
        key_body_pos_b = key_body_pos_b.view(n, -1)

        return {
            "root_pos_w":       root_pos_w,
            "root_quat":        root_quat,
            "root_vel_w":       root_vel_w,
            "root_vel_b":       root_vel_b,
            "root_ang_vel_w":   root_ang_vel_w,
            "root_ang_vel_b":   root_ang_vel_b,
            "dof_pos":          dof_pos,
            "dof_vel":          dof_vel,
            "key_body_pos_b":   key_body_pos_b,
        }

    def get_amp_observation(
        self, motion_ids: torch.Tensor, motion_times: torch.Tensor
    ) -> torch.Tensor:
        """Return expert AMP observations with the same 43-D layout as go2_amp.

        Matches AMPLoader.feed_forward_generator():
            frame[JOINT_POSE_START_IDX:JOINT_VEL_END_IDX] + root_z

        Layout:
            dof_pos(12), key_body_pos_b(12), root_vel_b(3),
            root_ang_vel_b(3), dof_vel(12), root_z(1)
        """
        state = self.get_motion_state(motion_ids, motion_times)
        root_z = state["root_pos_w"][:, 2:3]
        return torch.cat(
            [
                state["dof_pos"],
                state["key_body_pos_b"],
                state["root_vel_b"],
                state["root_ang_vel_b"],
                state["dof_vel"],
                root_z,
            ],
            dim=-1,
        )

    def sample_amp_transitions(self, n: int, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample expert transition pairs ``(s, s_next)`` for the discriminator."""
        motion_ids = self.sample_motions(n)
        times = self.sample_transition_times(motion_ids, dt)
        return (
            self.get_amp_observation(motion_ids, times),
            self.get_amp_observation(motion_ids, times + dt),
        )

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int, dt: float | None = None
    ):
        """Yield expert AMP transition mini-batches, compatible with AMPLoader."""
        if dt is None:
            if self.time_between_frames is None:
                raise ValueError(
                    "dt must be provided when cfg.time_between_frames is None."
                )
            dt = self.time_between_frames
        for _ in range(num_mini_batch):
            yield self.sample_amp_transitions(mini_batch_size, dt)

class MotionDataManager(ManagerBase):
    """Manager that holds one or more MotionDataTerm instances.

    Each term manages a group of AMP reference motion data loaded from disk.
    The AnimationManager / discriminator runner retrieve a specific term by name.

    对应:
        - env.amp_loader (go2_amp.py)   → MotionDataTerm
        - runner 中的 amp_data (on_policy_runner.py) → 同一个或另一个 MotionDataTerm
    """

    _env: ManagerBasedEnv
    _terms: dict[str, MotionDataTerm]
    _term_cfgs: dict[str, MotionDataTermCfg]

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        if cfg is None:
            raise ValueError("MotionDataManager requires a valid configuration.")

        self._terms = {}
        self._term_cfgs = {}
        super().__init__(cfg, env)

    def __str__(self) -> str:
        msg = f"<MotionDataManager> contains {len(self._terms)} active terms.\n"

        table = PrettyTable()
        table.title = "Motion Data Manager Terms"
        table.field_names = ["Index", "Term Name", "Total Duration", "Num Motions"]
        table.align["Term Name"] = "l"
        table.align["Total Duration"] = "r"
        table.align["Num Motions"] = "r"

        for idx, (name, term) in enumerate(self._terms.items()):
            table.add_row([
                idx, name,
                f"{term.get_total_duration():.1f} s",
                term.get_num_motions(),
            ])
        msg += table.get_string() + "\n"
        return msg

    @property
    def active_terms(self) -> list[str]:
        return list(self._terms.keys())

    def get_term(self, term_name: str) -> MotionDataTerm:
        if term_name not in self._terms:
            raise KeyError(
                f"Motion data term '{term_name}' not found. "
                f"Available: {list(self._terms.keys())}"
            )
        return self._terms[term_name]

    def _prepare_terms(self) -> None:
        cfg_items = (
            self.cfg.items() if isinstance(self.cfg, dict)
            else self.cfg.__dict__.items()
        )
        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, MotionDataTermCfg):
                raise TypeError(
                    f"Term '{term_name}' is not MotionDataTermCfg. "
                    f"Got: {type(term_cfg)}"
                )
            self._terms[term_name] = MotionDataTerm(term_cfg, self._env)
            self._term_cfgs[term_name] = term_cfg
