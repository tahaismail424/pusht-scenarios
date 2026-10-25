# Task success for teleop: is the block on the drawn goal outline?
#
# swm's PushT.eval_state compares against goal_state, a sampled agent+block
# state that includes the agent's xy, so it isn't "block on the outline". The
# outline is drawn with the block's own shapes at goal_pose, so the block sits
# exactly on it when block.position == goal_pose[:2] and the angles match
# (modulo the shape's rotational symmetry).
import numpy as np


def block_goal_error(env) -> tuple[float, float]:
    """Return (position error in px, angle error in rad) of block vs goal_pose."""
    base = env.unwrapped
    goal = np.asarray(base.goal_pose, dtype=np.float64)
    pos_err = float(np.linalg.norm(np.asarray(base.block.position) - goal[:2]))

    shape = base.shapes[int(base.variation_space['block']['shape'].value)]
    symmetry = base.shape_symmetry_angles.get(shape, 2 * np.pi)
    angle_err = abs(base.block.angle - goal[2]) % symmetry
    angle_err = float(min(angle_err, symmetry - angle_err))
    return pos_err, angle_err


def block_on_goal(env, pos_tol=20.0, angle_tol=np.deg2rad(20)) -> bool:
    """swm's tolerances by default: within 20 px and 20 degrees."""
    pos_err, angle_err = block_goal_error(env)
    return pos_err < pos_tol and angle_err < angle_tol
