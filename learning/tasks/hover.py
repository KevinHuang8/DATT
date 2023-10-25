import numpy as np

from gym import Env, spaces

from DATT.quadsim.sim import QuadSim
from DATT.quadsim.models import IdentityModel
from DATT.quadsim.rigid_body import State
from DATT.learning.base_env import BaseQuadsimEnv
from DATT.learning.configuration.configuration import AllConfig


class HoverEnv(BaseQuadsimEnv):
  """
     state is pos, vel, quat
     action is u, angvel
  """
  def reward(self, state, action):
    yaw = state.rot.as_euler('ZYX')[0]
    yawcost = 0.5 * min(abs(0 - yaw), abs(0 - yaw))
    poscost = min(np.linalg.norm(state.pos), 1.0)
    velcost = 0.2 * min(np.linalg.norm(state.vel), 1.0)

    ucost = 0.2 * abs(action[0]) + 0.1 * np.linalg.norm(action[1:])
    # had the u cost for old versions of hover by accident!
    cost = yawcost + poscost + velcost

    return -cost
