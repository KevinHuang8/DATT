import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from DATT.quadsim.sim import QuadSim
from DATT.quadsim.cascaded import CascadedController
from DATT.quadsim.cascaded_angvel import CascadedControllerAngvel
from DATT.quadsim.flatref import StaticRef, PosLine
from DATT.quadsim.models import IdentityModel
from DATT.quadsim.rigid_body import State_struct
from DATT.quadsim.visualizer import Vis

from DATT.python_utils.plotu import subplot, set_3daxes_equal

import DATT.quadsim.rot_metrics as rot_metrics

if __name__ == "__main__":
  posdes = np.array((1, 1, 0.5))
  yawdes = 0 * np.pi / 2
  dt = 0.005
  t_end = 3.0
  vis = True
  plot = True

  ref = StaticRef(pos=posdes, yaw=yawdes)
  #ref = PosLine(start = np.array((0, 0, 0)), end=posdes, yaw=yawdes, duration=2)

  model = IdentityModel()
  quadsim = QuadSim(model, vis=vis)

  controller = CascadedController(model, rot_metric=rot_metrics.rotvec_tilt_priority2)
  controller_angvel = CascadedControllerAngvel(model, rot_metric=rot_metrics.rotvec_tilt_priority2)

  ctrls = [
      (controller, "ang acc"),
      (controller_angvel, "ang vel"),
  ]

  tss = []
  for ctrl, lab in ctrls:
    ctrl.ref = ref
    quadsim.setstate(State_struct())
    ts = quadsim.simulate(dt=dt, t_end=t_end, controller=ctrl)

    tss.append((ts, lab))

  if not plot:
    sys.exit(0)

  for ts, lab in tss:
    eulers = np.array([rot.as_euler('ZYX')[::-1] for rot in ts.rot])
    eulers_des = np.array([rot.as_euler('ZYX')[::-1] for rot in ts.rotdes])

    subplot(ts.times, ts.pos, yname="Pos. (m)", title="Position", label=lab)
    subplot(ts.times, ts.vel, yname="Vel. (m)", title="Velocity", label=lab)
    subplot(ts.times, eulers, yname="Euler (rad)", title="ZYX Euler Angles", label=lab)
    subplot(ts.times, eulers_des, yname="Euler (rad)", title="ZYX Euler Angles", label=lab + "(des)")
    subplot(ts.times, ts.ang, yname="$\\omega$ (rad/s)", title="Angular Velocity", label=lab)
    subplot(ts.times, ts.angvel, yname="$\\omega$ (rad/s)", title="Angular Velocity", label=lab + "(des)")

    fig = plt.figure(num="Trajectory")
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(ts.pos[:, 0], ts.pos[:, 1], ts.pos[:, 2])
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    plt.title("Trajectory")
    set_3daxes_equal(ax)

  plt.show()
