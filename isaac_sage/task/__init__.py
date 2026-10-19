"""G1 cooperative-transport task (three G1 robots carrying one board).

Importing this package registers ``Isaac-G1-CooperativeTransport-Direct-v0`` with
gymnasium.  It imports Isaac Lab, so import it only after the Isaac Sim app has
been launched (``isaac_env.register_task`` does this).
"""

import gymnasium as gym

TASK_ID = "Isaac-G1-CooperativeTransport-Direct-v0"

gym.register(
    id=TASK_ID,
    entry_point=f"{__name__}.cooperative_transport_env:G1CooperativeTransportEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cooperative_transport_env:G1CooperativeTransportEnvCfg",
    },
)
