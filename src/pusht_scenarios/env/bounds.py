# Keep the agent on the board, under any physics.
#
# swm's default agent is kinematic and passes straight through the walls (it
# can end up far off the 512 px board). Rather than changing the physics, we
# bound the *action*: the env sets the PD target to agent + action * 100 px,
# and the PD gains are critically damped (k_v = 2 * sqrt(k_p)), so the agent
# approaches its target without overshoot. A target kept inside the walls
# therefore keeps the agent inside, for a kinematic or a dynamic agent alike.
#
# Bounding the action (not clamping inside the env) keeps the env identical to
# swm, and means recorded actions stay valid when replayed in plain swm. Record
# the bounded action: it is returned in info['applied_action'].
import gymnasium as gym
import numpy as np
import pymunk

WALL_INNER = (7.0, 504.0)  # inner faces of swm's walls (segments at 5/506, radius 2)


def agent_radius(base) -> float:
    """Largest extent of the agent's shapes from its body origin (circle: its
    radius). Uses local geometry: shape.bb is stale until the next space step."""
    extent = 0.0
    for shape in base.agent.shapes:
        if isinstance(shape, pymunk.Circle):
            extent = max(extent, shape.offset.length + shape.radius)
        elif isinstance(shape, pymunk.Poly):
            extent = max(extent, max(v.length for v in shape.get_vertices()) + shape.radius)
    return extent


def bound_action(base, action, margin: float = 1.0) -> np.ndarray:
    """Shrink a relative action so its PD target keeps the agent on the board."""
    action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
    r = agent_radius(base) + margin
    lo, hi = WALL_INNER[0] + r, WALL_INNER[1] - r
    agent = np.asarray(base.agent.position, dtype=np.float64)
    target = np.clip(agent + action * base.action_scale, lo, hi)
    # still within the action space if the agent was already out of bounds
    return np.clip((target - agent) / base.action_scale, -1, 1).astype(np.float32)


class KeepAgentOnBoard(gym.Wrapper):
    """Bounds every action (see bound_action); info['applied_action'] is what was applied."""

    def __init__(self, env, margin: float = 1.0):
        super().__init__(env)
        self.margin = margin

    def step(self, action):
        applied = bound_action(self.env.unwrapped, action, self.margin)
        obs, reward, terminated, truncated, info = self.env.step(applied)
        info = dict(info, applied_action=applied)  # swm's wrappers own info['action']
        return obs, reward, terminated, truncated, info


def make_bounded_pusht(**kwargs):
    """Entry point of pts/PushTBounded-v0: PushT with the agent kept on the
    board, for planners (CEM etc.) whose actions aren't bounded."""
    from pusht_scenarios.env.env import PushT

    return KeepAgentOnBoard(PushT(**kwargs))
