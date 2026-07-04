"""Minimal quaternion helpers for translating MuJoCo state to policy inputs."""

from __future__ import annotations

import numpy as np


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """Return a normalized [w, x, y, z] quaternion."""
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert a [w, x, y, z] quaternion to a body-to-world rotation matrix."""
    w, x, y, z = normalize_quat_wxyz(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotate_world_to_body(quat_wxyz: np.ndarray, vec_world: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the body frame."""
    rot_body_to_world = quat_wxyz_to_matrix(quat_wxyz)
    return rot_body_to_world.T @ np.asarray(vec_world, dtype=np.float64)


def projected_gravity_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Return gravity direction expressed in the base frame.

    IsaacLab's projected_gravity observation is the world gravity direction
    rotated into the robot base frame. The vector is unit length and points
    downward, so flat upright robots see approximately [0, 0, -1].
    """
    return rotate_world_to_body(quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float64))
