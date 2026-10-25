from gymnasium.envs.registration import register

register(
    id="pts/PushT-v0",
    entry_point="pusht_scenarios.env.env:PushT"
)