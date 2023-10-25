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

from DATT.learning.tasks.trajectory_fbff import TrajectoryEnv

class TrajectoryYawEnv(TrajectoryEnv):
  """
     state is pos, vel, quat
     action is u, angvel
  """
  def __init__(self, ref=None, y_max=0.0, seed=None, include_attitude_cost=False, fixed_seed=False, reset_freq=1, reset_thresh=0, relative=False, *args, **kwargs):
    self.time_horizon = 10
    self.include_attitude_cost = include_attitude_cost
    self.seed = seed
    self.reset_count = 0
    self.reset_freq = reset_freq
    self.relative = relative
    self.reset_thresh = reset_thresh

    if ref is None:
        self.ref = LineRef(D=1.0, altitude=0.0, period=1)
    else:
      self.ref = ref.ref(y_max=y_max, seed=self.seed, fixed_seed=fixed_seed)
    self.dt = 0.02
    BaseQuadsimEnv.__init__(self, *args, **kwargs)

    #self.ref = CircleRef(rad=1, altitude=0.0)
    self.action_space = spaces.Box(low=np.array([-9.8, -20, -20, -2]), high=np.array([30, 20, 20, 2]))

    self.all_mins = np.r_[self.all_mins, -50 * np.ones(4 * (self.time_horizon + 1))]
    self.all_maxes = np.r_[self.all_maxes, 50*np.ones(4 * (self.time_horizon + 1))]
    self.observation_shape = (self.observation_shape[0] + 10,)
    self.observation_space = spaces.Box(low=self.all_mins, high=self.all_maxes)

  def yawdiff(self, curr, des):
    return np.arctan2(np.sin(curr - des), np.cos(curr - des))

  def obs(self, state):
    obs_ = BaseQuadsimEnv.obs(self, state)
    rot = R.from_quat(obs_[6:10])
    yaw = state.rot.as_euler('ZYX')[0]
    if self.relative:
      # assume body frame = True
      fb = obs_[0:3] - rot.inv().apply(self.ref.pos(self.t))
      fb = np.r_[fb, self.yawdiff(yaw, self.ref.yaw(self.t))]
      obs_ = np.hstack(
            [obs_, fb] 
            + [np.r_[obs_[0:3] - rot.inv().apply(self.ref.pos(self.t + 3 * i * self.dt)), self.yawdiff(yaw, self.ref.yaw(self.t + 3 * i * self.dt))] for i in range(self.time_horizon)]
        )
    else:
      if self.body_frame:
        fb = obs_[0:3] - rot.inv().apply(self.ref.pos(self.t))
      else:
        fb = obs_[0:3] - self.ref.pos(self.t)
      fb = np.r_[fb, self.yawdiff(yaw, self.ref.yaw(self.t))]
      obs_ = np.hstack([obs_, fb] + [np.r_[self.ref.pos(self.t + 3 * i * self.dt), self.ref.yaw(self.t + 3 * i * self.dt)] for i in range(self.time_horizon)])
    return obs_

  def reward(self, state, action):
    yaw = state.rot.as_euler('ZYX')[0]
    yawcost = 0.5 * np.abs(self.yawdiff(yaw, self.ref.yaw(self.t)))
    poscost = np.linalg.norm(state.pos - self.ref.pos(self.t))#min(np.linalg.norm(state.pos), 1.0)
    velcost = 0.1 * min(np.linalg.norm(state.vel), 1.0)
    # cosine distance for attitude
    costheta = state.rot.as_matrix()[2, 2]
    attitude_cost = 0.1 * (1 - costheta)

    ucost = 0.2 * abs(action[0]) + 0.1 * np.linalg.norm(action[1:])

    if self.include_attitude_cost:
      cost = yawcost + poscost + velcost + attitude_cost
    else:
      cost = yawcost + poscost + velcost

    return -cost
