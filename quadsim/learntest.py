import copy

import numpy as np

from scipy.spatial.transform import Rotation as R

from quadsim.cascaded import CascadedController, CascadedControllerLearnAccel
from quadsim.dist import MassDisturbance, LinearDrag, InertiaDisturbance, MotorModelDisturbance, WindField
from quadsim.fblin import FBLinController, FBLinControllerLearnAccel
from quadsim.flatref import StaticRef, PosLineYawLine, YawLine, PosLine
from quadsim.learn import InputVel, InputPos, InputPosVel
from quadsim.models import IdentityModel, rocky09
from quadsim.rigid_body import State

from quadsim.compare import Test, plot, run

import quadsim.rot_metrics as rot_metrics

from regression import Linear, SSGPR

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

if __name__ == "__main__":
  startpos = np.zeros(3)
  endpos = np.array((0, 3, 0.0))
  startyaw = 0.0
  #endyaw = np.pi / 2
  endyaw = 0.0
  duration = 2.0

  startstate = State(
    pos=startpos,
    vel=np.zeros(3),
    rot=R.from_euler('ZYX', [startyaw, 0.0, 0.0]),
    ang=np.zeros(3)
  )

  dt = 0.005
  t_end = 2.0

  n_trials = 8

  sim_motors = True

  #ref = StaticRef(pos=endpos, yaw=endyaw)
  #ref = PosLine(start=startpos, end=endpos, yaw=endyaw, duration=duration)
  #ref = YawLine(pos=endpos, startyaw=startyaw, endyaw=endyaw, duration=duration)
  ref = PosLineYawLine(start=startpos, end=endpos, startyaw=startyaw, endyaw=endyaw, duration=duration)

  dists = [
    #MassDisturbance(1.2),
    #InertiaDisturbance((1.3, 1.2, 1.5)),
    #LinearDrag(0.6),
    WindField(pos=np.array((-1, 1.5, 0.0)), direction=np.array((1, 0, 0)), noisevar=0.0, vmax=100.0, decay_long=1.8),
    #MotorModelDisturbance(0.8),
  ]

  model_control = rocky09()
  #model_control = IdentityModel()

  model_true = copy.deepcopy(model_control)

  #learner = Linear()
  #features = InputPosVel()
  features = InputPos()
  learner = SSGPR(N_feats=100, lengths=0.3 * np.ones(features.dim), include_constant=True, include_linear=True)

  def casc(rm):
    return CascadedController(model_control, rot_metric=rm)

  fblin_base = FBLinController(model_control, dt=dt)
  fblin_learn = FBLinControllerLearnAccel(model_control, learner, features, dt=dt)

  tests = [
    #Test(casc(rot_metrics.rotvec_tilt_priority2), label="Baseline FF"),
    Test(CascadedControllerLearnAccel(model_control, learner, features, rot_metric=rot_metrics.rotvec_tilt_priority2), label="FFLin Learn", n_trials=n_trials),
    #Test(fblin_base, label="Baseline FB"),
    Test(fblin_learn, label="FBLin Learn", n_trials=n_trials)
  ]

  run(model_true, startstate, ref, dists, tests, dt, t_end, sim_motors=sim_motors)
  plot(tests, ref)
