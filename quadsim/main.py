import sys

import numpy as np

from DATT.quadsim.sim import QuadSim
from DATT.quadsim.cascaded import CascadedController
from DATT.quadsim.fblin import FBLinController
from DATT.quadsim.flatref import StaticRef, PosLine
# from DATT.quadsim.pid_controller import PIDController
# from DATT.learning.expert_pid_controller_trajectory import PIDController as PIDControllerTrajectory
from DATT.quadsim.flatref import StaticRef, PosLine
from DATT.quadsim.models import IdentityModel
from DATT.quadsim.dist import WindField, ConstantForce
from DATT.quadsim.circleref import CircleRef
from DATT.quadsim.lineref import LineRef
from DATT.learning.refs.square_ref import SquareRef
from DATT.learning.refs.random_zigzag import RandomZigzag
from DATT.learning.refs.pointed_star import NPointedStar
# from DATT.learning.policy_controller import PolicyController
from DATT.learning.refs.gen_trajectory import main_loop, Trajectory
from DATT.quadsim.fig8ref import Fig8Ref


from DATT.quadsim.controllers.pid_controller import PIDController
from DATT.quadsim.controllers.mppi_controller import MPPIController


from DATT.python_utils.plotu import subplot, set_3daxes_equal
from DATT.quadsim.controllers.cntrl_config import PIDConfig, MPPIConfig
import DATT.quadsim.rot_metrics as rot_metrics


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

  cntrl_config = MPPIConfig()
  controller = MPPIController(model, cntrl_config=cntrl_config)

  # cntrl_config = PIDConfig()
  # controller = PIDController(model, cntrl_config=cntrl_config)
  
  controller.ref_func = ref
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
