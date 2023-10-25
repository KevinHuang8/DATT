import sys

import numpy as np

from quadsim.sim import QuadSim
from quadsim.cascaded import CascadedController
from quadsim.fblin import FBLinController
from quadsim.flatref import StaticRef, PosLine
from quadsim.pid_controller import PIDController
from quadsim.learning.expert_pid_controller_trajectory import PIDController as PIDControllerTrajectory
from quadsim.flatref import StaticRef, PosLine
from quadsim.models import IdentityModel
from quadsim.dist import WindField, ConstantForce
from quadsim.circleref import CircleRef
from quadsim.lineref import LineRef
from quadsim.learning.refs.square_ref import SquareRef
from quadsim.learning.refs.random_zigzag import RandomZigzag
from quadsim.learning.refs.pointed_star import NPointedStar
from quadsim.learning.policy_controller import PolicyController
from quadsim.learning.refs.gen_trajectory import main_loop, Trajectory
from quadsim.fig8ref import Fig8Ref

from python_utils.plotu import subplot, set_3daxes_equal

import quadsim.rot_metrics as rot_metrics

import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":
  import argparse
  import time

  parser = argparse.ArgumentParser()
  parser.add_argument('--policyname', '-n', default='hover_basic', type=str, help='Policy name to load')

  args = parser.parse_args()

  posdes = np.array((1.0, 1.0, 1.0))
  # yawdes = np.pi / 2
  yawdes = 0.0
  dt = 0.02
  vis = True
  plot = True

  t_end = 25.0

  ref = NPointedStar(n_points=5, speed=2, radius=1)
  # ref = main_loop(saved_traj='test_ref', parent=Path().absolute() / 'learning' / 'refs')

  model = IdentityModel()

  quadsim = QuadSim(model, vis=vis)
  #controller = CascadedController(model, rot_metric=rot_metrics.rotvec_tilt_priority2)
  # controller = PolicyController(model, algoname='ppo', policyname=args.policyname)
  # controller = CascadedController(model, rot_metric=rot_metrics.euler_zyx)
  #controller = FBLinController(model, dt=dt)
  controller = PIDController(model)

  controller.ref = ref
  dists = [
    # ConstantForce(np.array([4, 4, 4]))
    # WindField(pos=np.array((-1, 1.5, 0.0)), direction=np.array((1, 0, 0)), noisevar=25.0, vmax=1500.0, decay_long=1.8)
  ]
  ts = quadsim.simulate(dt=dt, t_end=t_end, controller=controller, dists=dists)

  if not plot:
    sys.exit(0)

  eulers = np.array([rot.as_euler('ZYX')[::-1] for rot in ts.rot])

  plt.figure()
  ax = plt.subplot(3, 1, 1)
  plt.plot(ts.times, ts.pos[:, 0])
  plt.plot(ts.times, ref.pos(ts.times)[0, :])
  plt.subplot(3, 1, 2, sharex=ax)
  plt.plot(ts.times, ts.pos[:, 1])
  plt.plot(ts.times, ref.pos(ts.times)[1, :])
  plt.subplot(3, 1, 3, sharex=ax)
  plt.plot(ts.times, ts.pos[:, 2])
  plt.plot(ts.times, ref.pos(ts.times)[2, :])
  plt.suptitle(type(controller).__name__)

  plt.figure()

  plt.plot(ts.pos[:, 0], ts.pos[:, 1], label='actual')
  # plt.plot(ref.pos(ts.times)[0, :], ref.pos(ts.times)[1, :], label='desired')
  plt.legend()

  # subplot(ts.times, ts.pos, yname="Pos. (m)", title="Position", des=ref.pos(ts.times))
  subplot(ts.times, ts.vel, yname="Vel. (m)", title="Velocity")

  subplot(ts.times, ref.vel(ts.times).T, yname="Vel. (m)", title="Velocity", label="Desired")

  subplot(ts.times, eulers, yname="Euler (rad)", title="ZYX Euler Angles")
  subplot(ts.times, ts.ang, yname="$\\omega$ (rad/s)", title="Angular Velocity")
  subplot(ts.times, ts.force, yname="Force (N)", title="Body Z Thrust")

  # fig = plt.figure(num="Trajectory")
  # ax = fig.add_subplot(111, projection='3d')
  # plt.plot(ts.pos[:, 0], ts.pos[:, 1], ts.pos[:, 2])
  # plt.xlabel("X (m)")
  # plt.ylabel("Y (m)")
  # ax.set_zlabel("Z (m)")
  # plt.title("Trajectory")
  # set_3daxes_equal(ax)

  plt.show()
