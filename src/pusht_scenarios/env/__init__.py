from gymnasium.envs.registration import register

register(
    id="pts/PushT-v0",
    entry_point="pusht_scenarios.env.env:PushT"
)

# Same env, actions bounded so the agent can't leave the board (env/bounds.py).
# Use this id for evaluation/planning; recorded and augmented data already
# store bounded actions.
register(
    id="pts/PushTBounded-v0",
    entry_point="pusht_scenarios.env.bounds:make_bounded_pusht"
)
