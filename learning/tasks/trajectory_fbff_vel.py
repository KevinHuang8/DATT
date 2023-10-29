import numpy as np

from gym import Env, spaces

from DATT.quadsim.sim import QuadSim
from DATT.quadsim.models import IdentityModel
from DATT.quadsim.rigid_body import State_struct
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
  def __init__(self, ref=None, *args, **kwargs):
    self.time_horizon = 10

    if ref is None:
      self.ref = LineRef(D=1.0, altitude=0.0, period=1)
    else:
      self.ref = ref.ref() #LineRef(D=1.0, altitude=0.0, period=1)
    self.dt = 0.02
    super().__init__(*args, **kwargs)

    #self.ref = CircleRef(rad=1, altitude=0.0)
    self.action_space = spaces.Box(low=np.array([-9.8, -20, -20, -2]), high=np.array([30, 20, 20, 2]))

    self.all_mins = np.r_[self.all_mins, -50 * np.ones(6 * (self.time_horizon) + 3)]
    self.all_maxes = np.r_[self.all_maxes, 50*np.ones(6 * (self.time_horizon) + 3)]
    self.observation_shape = (self.observation_shape[0] + 10,)
    self.observation_space = spaces.Box(low=self.all_mins, high=self.all_maxes)

  def reset(self, state=None):
    super().reset(state)

    if state is None:
        state = State_struct(
            pos=np.zeros(3),
            vel=np.zeros(3),
            rot=R.from_euler('ZYX', np.zeros(3), degrees=True),
            ang=np.zeros(3)
        )

    self.quadsim.setstate(state)
    return self.obs(state)

  def obs(self, state):
    obs_ = super().obs(state)
    ff_pos = np.array([self.ref.pos(self.t + 3 * i * self.dt) for i in range(self.time_horizon)])
    ff_pos_stacked = ff_pos.reshape(-1, 3)
    ff_vel = np.diff(ff_pos_stacked, axis=0) / (3 * self.dt)
    # pad last timestep by repeating previous value
    ff_vel = np.r_[ff_vel, ff_vel[-1, None]]
    ff_terms_stacked = np.c_[ff_pos, ff_vel]
    ff_terms = ff_terms_stacked.flatten()

    obs_ = np.hstack([obs_, obs_[0:3] - self.ref.pos(self.t), ff_terms])
    # obs_ = np.hstack((obs_, self.ref.pos(self.t), self.ref.pos(self.t + self.dt), self.ref.pos(self.t + self.dt*2), self.ref.yaw(self.t)))
    return obs_

  def reward(self, state, action):
    yaw = state.rot.as_euler('ZYX')[0]
    yawcost = 0.5 * min(abs(self.ref.yaw(self.t) - yaw), abs(self.ref.yaw(self.t) - yaw))
    poscost = np.linalg.norm(state.pos - self.ref.pos(self.t))#min(np.linalg.norm(state.pos), 1.0)

    ref_vel = (self.ref.pos(self.t + 3 * self.dt) - self.ref.pos(self.t)) / (3 * self.dt)
    velcost = 0.1 * min(np.linalg.norm(state.vel), 1.0)#np.linalg.norm(state.vel - ref_vel) # not properly scaled

    ucost = 0.2 * abs(action[0]) + 0.1 * np.linalg.norm(action[1:])

    cost = yawcost + poscost + velcost

    return -cost
