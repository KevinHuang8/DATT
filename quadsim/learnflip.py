import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from quadsim.sim import QuadSim
from quadsim.cascaded import CascadedController
from quadsim.cascaded_angvel import CascadedControllerAngvel
from quadsim.fblin import FBLinController
from quadsim.flatref import StaticRef, PosLine
from quadsim.models import IdentityModel
from quadsim.rigid_body import State
from quadsim.visualizer import Vis

from python_utils.plotu import subplot, set_3daxes_equal

import quadsim.rot_metrics as rot_metrics

from gym import Env, spaces

class FlipEnv(Env):
  """
     state is pos, vel, quat
     action is u, angvel
  """
  def __init__(self):
    self.dt = 0.02

    model = IdentityModel()
    self.quadsim = QuadSim(model, vis=vis)
    self.reset()

    self.observation_shape = (10,)
    self.observation_space = spaces.Box(low=-2 * np.ones(10), high=2 * np.ones(10))

    self.action_shape = (4,)
    self.action_space = spaces.Box(low=-20 * np.ones(4), high=20 * np.ones(4))

  def reset(self, state=None):
    # TODO Add randomization here for motor delays and noise, etc.
    if state is None:
      state = State(
                pos=np.random.uniform(low=-0.5, high=0.5, size=(3,)),
                vel=np.random.uniform(low=-0.5, high=0.5, size=(3,))
              )

    self.t = 0
    self.quadsim.setstate(state)
    return self.obs(state)

  def obs(self, state):
    return np.hstack((state.pos, state.vel, state.rot.as_quat()))

  def step(self, action):
    u, angvel = action[0], action[1:]

    u += 9.8

    state = self.quadsim.step_angvel_raw(self.dt, u, angvel)
    self.t += self.dt

    failed = np.linalg.norm(state.pos) > 1.5

    o = self.obs(state)
    r = self.reward(state, action)
    done = self.t >= 10.0 or failed
    info = dict()

    if failed:
      r -= 10000

    return o, r, done, info

  def reward(self, state, action):
    yaw = state.rot.as_euler('ZYX')[0]
    yawcost = 0.5 * min(abs(np.pi - yaw), abs(-np.pi - yaw))
    poscost = min(np.linalg.norm(state.pos), 1.0)
    velcost = 0.2 * min(np.linalg.norm(state.vel), 1.0)

    ucost = 0.2 * abs(action[0]) + 0.1 * np.linalg.norm(action[1:])

    cost = yawcost + poscost + velcost + ucost

    return -cost

class LCAngvel:
  def __init__(self):
    pass

  def response(self, t, state):
    return 0, 0

if __name__ == "__main__":
  posdes = np.array((1, 1, 0.5))
  yawdes = 0 * np.pi / 2
  dt = 0.005
  vis = False
  plot = True

  retrain = False
  moretrain = True
  traints = 2e6
  moretraints = 1e6

  checkenv = False

  ref = StaticRef(pos=posdes, yaw=yawdes)
  #ref = PosLine(start = np.array((0, 0, 0)), end=posdes, yaw=yawdes, duration=2)

  from stable_baselines3 import A2C, PPO
  from stable_baselines3.common.evaluation import evaluate_policy
  from stable_baselines3.common.env_checker import check_env
  from stable_baselines3.common.env_util import make_vec_env

  if checkenv:
    # For some reason this fails?
    env = FlipEnv()
    obs = env.reset()
    print(obs)
    check_env(env)

  #env = FlipEnv()
  env = make_vec_env(FlipEnv, n_envs=8)

  #pname = "ppo_flip111"
  pname = "ppo_flip4"

  pc = PPO

  if not Path(f"{pname}.zip").exists() or retrain:
    policy = pc("MlpPolicy", env, tensorboard_log='./logs/learn_flip_ppo/', verbose=1)

    mean_r, std_r = evaluate_policy(policy, env, n_eval_episodes=10)
    print(mean_r, std_r)

    policy.learn(total_timesteps=traints, progress_bar=True)
    policy.save(pname)
  else:
    policy = pc.load(pname, env)

    if moretrain:
      policy.set_env(env)
      policy.learn(total_timesteps=moretraints, progress_bar=True)
      policy.save(f"{pname}1")

  obs = env.reset()
  mean_r, std_r = evaluate_policy(policy, env, n_eval_episodes=10)
  print(mean_r, std_r)

  evalenv = FlipEnv()
  evalstate = State(pos=np.array((0.0, 0.0, 0.0)), vel=np.array((-0.0, 0, 0.)),)
  obs = evalenv.reset(evalstate)
  vis = Vis()
  total_r = 0
  for i in range(50000):
    action, _states = policy.predict(obs)
    obs, rewards, dones, info = evalenv.step(action)

    total_r += rewards

    state = evalenv.quadsim.rb.state()
    vis.set_state(state.pos.copy(), state.rot)
    if i < 50:
      time.sleep(0.08)

  print(total_r)
