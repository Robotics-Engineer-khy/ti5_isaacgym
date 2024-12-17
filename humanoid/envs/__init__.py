


from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
# print(f"Actor MLP: {self.actor}")
# print(f"Critic MLP: {self.critic}")
# print(f"long_history CNN: {self.long_history}")
# print(f"state_estimator MLP: {self.state_estimator}")
from .base.legged_robot import LeggedRobot
from .t1.t1_dh_stand_config import DHT1StandCfg, DHT1StandCfgPPO
from .t1.t1_dh_stand_env import T1DHStandEnv
from humanoid.utils.task_registry import task_registry

task_registry.register( "t1_dh_stand", T1DHStandEnv, DHT1StandCfg(), DHT1StandCfgPPO() )

