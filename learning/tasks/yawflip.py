import numpy as np

from gym import spaces

from DATT.learning.base_env import BaseQuadsimEnv
from DATT.learning.configuration.configuration import AllConfig

class YawflipEnv(BaseQuadsimEnv):
  """
     state is pos, vel, quat
     action is u, angvel
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.action_space = spaces.Box(low=np.array([-9.8, -20, -20, -2]), high=np.array([30, 20, 20, 2]))
    self.t_end = 5.0
    self.pos_weight = 1.0
    self.yawdes = np.pi

  def reward(self, state, action):
    yaw = state.rot.as_euler('ZYX')[0]

    yawcost = 0.5 * min(abs(self.yawdes - yaw), abs(-self.yawdes - yaw))
    poscost = self.pos_weight * min(np.linalg.norm(state.pos), 1.0)

    cost = yawcost + poscost

    return -cost

class YawflipEnvZeroYaw(YawflipEnv):
  """
     state is pos, vel, quat
     action is u, angvel
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.yawdes = 0
