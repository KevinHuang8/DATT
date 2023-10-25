import numpy as np

from gym import Env, spaces

from DATT.quadsim.sim import QuadSim
from DATT.quadsim.models import IdentityModel
from DATT.quadsim.rigid_body import State
from DATT.learning.base_env import BaseQuadsimEnv
from DATT.learning.configuration.configuration import AllConfig
from DATT.quadsim.circleref import CircleRef
from DATT.quadsim.lineref import LineRef
from scipy.spatial.transform import Rotation as R


class TrajectoryEnv(BaseQuadsimEnv):
  """
     state is pos, vel, quat
     action is u, angvel
  """
  def __init__(self, *args, **kwargs):
    self.ref = LineRef(D=1.0, altitude=0.0, period=1)
    self.dt = 0.02
    super().__init__(*args, **kwargs)

    #self.ref = CircleRef(rad=1, altitude=0.0)
    self.action_space = spaces.Box(low=np.array([-9.8, -20, -20, -2]), high=np.array([30, 20, 20, 2]))

    self.all_mins = np.r_[self.all_mins, -50 * np.ones(3), -2*np.pi]
    self.all_maxes = np.r_[self.all_maxes, 50*np.ones(3), 2*np.pi]
    self.observation_shape = (self.observation_shape[0] + 10,)
    self.observation_space = spaces.Box(low=self.all_mins, high=self.all_maxes)

  def reset(self, state=None):
    super().reset(state)

    if state is None:
        state = State(
            pos=np.zeros(3),
            vel=np.zeros(3),
            rot=R.from_euler('ZYX', np.zeros(3), degrees=True),
            ang=np.zeros(3)
        )

    self.quadsim.setstate(state)
    return self.obs(state)

  def obs(self, state):
    obs_ = super().obs(state)
    obs_ = np.hstack((obs_, self.ref.pos(self.t), self.ref.yaw(self.t)))
    return obs_

  def reward(self, state, action):
    yaw = state.rot.as_euler('ZYX')[0]
    yawcost = 0.5 * min(abs(self.ref.yaw(self.t) - yaw), abs(self.ref.yaw(self.t) - yaw))
    poscost = np.linalg.norm(state.pos - self.ref.pos(self.t))#min(np.linalg.norm(state.pos), 1.0)
    velcost = 0.1 * min(np.linalg.norm(state.vel), 1.0)

    ucost = 0.2 * abs(action[0]) + 0.1 * np.linalg.norm(action[1:])

    cost = yawcost + poscost + velcost

    return -cost
