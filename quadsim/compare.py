import argparse
import copy

import numpy as np

from scipy.spatial.transform import Rotation as R

from quadsim.cascaded import CascadedController, thrust_project_z, thrust_maintain_z
from quadsim.dist import MassDisturbance, LinearDrag, InertiaDisturbance, MotorModelDisturbance, WindField
from quadsim.fblin import (
  FBLinController,
  FBLinControllerThrustDelay,
  FBLinControllerThrustDelayNL,
  FBLinControllerFullDelay
)
from quadsim.fflin import (
  FFLinController,
)
from quadsim.flatref import StaticRef, PosLineYawLine, YawLine, PosLine
from quadsim.models import IdentityModel, rocky09
from quadsim.rigid_body import State
from quadsim.sim import QuadSim, QuadSimMotors

from python_utils.mathu import normang, smoothang
from python_utils.plotu import subplot, set_3daxes_equal

import quadsim.rot_metrics as rot_metrics

import matplotlib.pyplot as plt

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

class Test:
  def __init__(self, controller, n_trials=1, **plotargs):
    self.controller = controller
    self.plotargs = plotargs
    self.n_trials = n_trials
    self.results = []

def run(model, startstate, ref, dists, tests, dt, t_end, sim_motors=False, sim_params=None):
  if sim_params is None:
    sim_params = dict()

  for dist in dists:
    dist.apply(model)

  if sim_motors:
    quadsim = QuadSimMotors(model, **sim_params)
  else:
    quadsim = QuadSim(model, **sim_params)

  for test in tests:
    print(test.plotargs['label'])

    for i in range(test.n_trials):
      quadsim.setstate(startstate)
      test.controller.ref = ref

      if sim_motors:
        test.controller.output_rpm = True

      ts = quadsim.simulate(dt=dt, t_end=t_end, controller=test.controller, dists=dists)

      if not i:
        test.posdes = ref.pos(ts.times).T
        test.veldes = ref.vel(ts.times).T
        test.yawdes = ref.yaw(ts.times)

      ts.poserr = ts.pos - test.posdes
      ts.poserrnorm = np.linalg.norm(ts.poserr, axis=1)

      ts.yaw = np.array([rot.as_euler('ZYX')[0] for rot in ts.rot])
      ts.yawerr = normang(ts.yaw - test.yawdes)

      ax_errs = np.mean(np.abs(ts.poserr), axis=0)
      yaw_err = np.mean(np.abs(ts.yawerr))

      test.results.append(ts)
      test.controller.endtrial()

      print(f"\tTrial {i + 1} mean err (m): {np.mean(ts.poserrnorm):.3f}, XYZ: {ax_errs[0]:.3f}, {ax_errs[1]:.3f}, {ax_errs[2]:.3f}. Yaw (rad): {yaw_err:.3f}")

def plot(tests, ref, desargs=None):
  if desargs is None:
    desargs = {}

  fig = plt.figure(num="Trajectory")
  ax = fig.add_subplot(111, projection='3d')
  ax.set_title("Trajectory")
  ax.set_xlabel("X (m)")
  ax.set_ylabel("Y (m)")
  ax.set_zlabel("Z (m)")

  desargs = dict(label="Desired", linestyle='dashed', color='black', **desargs)

  for i, test in enumerate(tests):
    plotargs = test.plotargs

    # Use last trial... Temporary TODO
    ts = test.results[-1]

    if len(test.results) > 1:
      print(f"WARNING: {plotargs['label']} has {len(test.results)} trials. Only plotting last.")

    if not i:
      subplot(ts.times, test.posdes, yname="(m)", title="Position", **desargs)
      subplot(ts.times, test.veldes, yname="(m/s)", title="Velocity", **desargs)
      plt.figure("ZYX Euler Angles")
      plt.subplot(313)
      plt.plot(ts.times, test.yawdes, **desargs)

      subplot(ts.times, test.yawdes, yname="Yaw (rad)", title="Yaw", **desargs)

    # Pos Vel
    subplot(ts.times, ts.pos, yname="(m)", title="Position", **plotargs)
    subplot(ts.times, ts.vel, yname="(m/s)", title="Velocity", **plotargs)

    # Euler
    eulers = np.array([rot.as_euler('ZYX')[::-1] for rot in ts.rot])
    subplot(ts.times, eulers, yname="Euler (rad)", title="ZYX Euler Angles", **plotargs)

    subplot(ts.times, smoothang(eulers[:, 2]), yname="Yaw (rad)", title="Yaw", **plotargs)

    # Angvel Torque
    subplot(ts.times, ts.ang, yname="$\\omega$ (rad/s)", title="Angular Velocity", **plotargs)
    subplot(ts.times, ts.torque, yname="Torque (Nm)", title="Control Torque", **plotargs)
    if hasattr(ts, 'torquedes'):
      subplot(ts.times, ts.torquedes, yname="Torque (Nm)", title="Control Torque", **plotargs, linestyle='dashed')

    # Thrust
    subplot(ts.times, ts.force, yname="Thrust (N)", title="Control Thrust", **plotargs)
    if hasattr(ts, 'forcedes'):
      subplot(ts.times, ts.forcedes, yname="Thrust (N)", title="Control Thrust", **plotargs, linestyle='dashed')

    if hasattr(ts, 'uddot'):
      subplot(ts.times, ts.uddot, yname="u ddot (m/s$^4$)", title="U ddot", **plotargs)

    # Pos Err
    subplot(ts.times, ts.poserr,  yname="Pos. Err. (m)", title="Position Error Per Axis", **plotargs)
    subplot(ts.times, ts.poserrnorm, yname="Position Error (m)", title="Position Error", **plotargs)
    # Yaw Err
    subplot(ts.times, ts.yawerr, yname="Yaw Err. (rad)", title="Yaw Error", **plotargs)

    if hasattr(ts, 'accel_error_true'):
      assert hasattr(ts, 'accel_error_pred')
      subplot(ts.times, ts.accel_error_true, yname="Acc. Err. (m/s$^2$)", title="Accel Error Learning", **plotargs, linestyle='dashed')
      subplot(ts.times, ts.accel_error_pred, yname="Acc. Err. (m/s$^2$)", title="Accel Error Learning", **plotargs)

    # 3D Traj
    ax.plot(ts.pos[:, 0], ts.pos[:, 1], ts.pos[:, 2], **plotargs)
    set_3daxes_equal(ax)

  plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--no-plot', action='store_true', help="Do not display any graphs")
  args = parser.parse_args()

  edgesize = 2.0

  startpos = np.zeros(3)
  #endpos = np.array((edgesize, edgesize, 0.0))
  endpos = np.array((edgesize / 2, edgesize, 0.0))
  startyaw = 0.0
  endyaw = 1.9 * np.pi / 2
  #endyaw = 0.0
  duration = 2.0

  startstate = State(
    pos=startpos,
    vel=np.zeros(3),
    rot=R.from_euler('ZYX', [startyaw, 0.0, 0.0]),
    ang=np.zeros(3)
  )

  dt = 0.001
  t_end = duration
  sim_motors = False

  #ref = StaticRef(pos=endpos, yaw=endyaw)
  #ref = PosLine(start=startpos, end=endpos, yaw=endyaw, duration=duration)
  #ref = YawLine(pos=endpos, startyaw=startyaw, endyaw=endyaw, duration=duration)
  ref = PosLineYawLine(start=startpos, end=endpos, startyaw=startyaw, endyaw=endyaw, duration=duration)

  dists = [
    #MassDisturbance(1.2),
    InertiaDisturbance((1.3, 1.2, 1.5)),
    LinearDrag(1.2),
    #MotorModelDisturbance(0.8),
    #WindField(pos=np.zeros(3), direction=np.array((0, 1, 0)), vmax=30),
    #WindField(pos=endpos, direction=np.array((0, -1, 0)), vmax=30)
  ]

  #model_control = IdentityModel()
  model_control = rocky09()
  model_true = copy.deepcopy(model_control)

  def casc(rm, u_f=thrust_project_z):
    return CascadedController(model_control, rot_metric=rm, u_f=u_f)

  fblin = FBLinController(model_control, dt=dt)
  fblin_td = FBLinControllerThrustDelay(model_control, dt=dt)
  fblin_tdnl = FBLinControllerThrustDelayNL(model_control, dt=dt)
  fblin_fd = FBLinControllerFullDelay(model_control, dt=dt)

  fflin = FFLinController(model_control, dt=dt)

  tests = [
    #Test(casc(rot_metrics.euler_zyx), label="Euler ZYX"),
    #Test(fblin_fd, label="FBLin (FD)"),
    #Test(casc(rot_metrics.rotvec_tilt_priority2), label="Rotation Vector TP"),
    #Test(casc(rot_metrics.rotvec_tilt_priority2, thrust_project_z), label="Project Z"),
    #Test(casc(rot_metrics.rotvec_tilt_priority2, thrust_maintain_z), label="Maintain Z"),
    Test(casc(rot_metrics.rotvec_tilt_priority2), label="Cascaded"),
    Test(fblin, label="FBLin"),
    Test(fflin, label="FFLin"),
    #Test(fblin_td, label="FBLin (TD)"),
    #Test(fblin_tdnl, label="FBLin (TDNL)"),
  ]

  run(model_true, startstate, ref, dists, tests, dt, t_end, sim_motors=sim_motors)
  if not args.no_plot:
    plot(tests, ref)
