import sys
import time

import numpy as np

from scipy.spatial.transform import Rotation as R

from python_utils.keygrabber import KeyGrabber

from quadsim.sim import QuadSim
from quadsim.cascaded import CascadedController
from quadsim.learning.policy_controller import PolicyController
from quadsim.flatref import StaticRef
from quadsim.models import IdentityModel

import quadsim.rot_metrics as rot_metrics

keymap = dict(
    left='a',
    right='d',
    forward='w',
    backward='s',
    down='-',
    up='=',
    turnleft=',',
    turnright='.',
    reset='r',
    dec_movedist='[',
    inc_movedist=']',
    dec_turndist='n',
    inc_turndist='m',
    ratedown='o',
    rateup='p',
    help='h',
)

def printhelp():
  print("=== KEY BINDINGS FOR DRONE CONTROL ===")
  for desc, key in keymap.items():
    print(f"{desc}: {key}")

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--policyname', '-n', default='hover_basic', type=str, help='Policy name to load')
  parser.add_argument('--speedup', default=3, type=float, help='How many times to speed up sim, 1 = "realtime"')
  parser.add_argument('--posnoise', default=0.0, type=float, help='std. dev. of position noise')

  args = parser.parse_args()

  dt = 0.005
  vis = True

  model = IdentityModel()
  quadsim = QuadSim(model, vis=vis)

  startstate = quadsim.getstate()

  #controller = CascadedController(model, rot_metric=rot_metrics.rotvec_tilt_priority2)
  controller = PolicyController(model, algoname='ppo', policyname=args.policyname, posnoise_std=args.posnoise)

  def resetref():
    controller.ref = StaticRef(np.zeros(3))

  resetref()

  kg = KeyGrabber()
  printhelp()

  movedist = 0.25
  yawdist = np.pi / 8

  movedistinc = 0.05
  yawdistinc = np.pi / 16

  i = 0
  while 1:
    quadsim.step(i * dt, dt, controller)

    if vis:
      state = quadsim.rb.state()
      quadsim.vis.set_state(state.pos.copy(), state.rot)
      time.sleep(dt / args.speedup)

    i += 1

    chars = kg.read()
    if chars:
      for c in chars:
        bodymove = np.zeros(3)
        if c == keymap['forward']:
          bodymove[0] += movedist
        elif c == keymap['left']:
          bodymove[1] += movedist
        elif c == keymap['backward']:
          bodymove[0] -= movedist
        elif c == keymap['right']:
          bodymove[1] -= movedist
        elif c == keymap['up']:
          bodymove[2] += movedist
        elif c == keymap['down']:
          bodymove[2] -= movedist
        elif c == keymap['turnleft']:
          controller.ref.yawdes += yawdist
        elif c == keymap['turnright']:
          controller.ref.yawdes -= yawdist
        elif c == keymap['reset']:
          quadsim.setstate(startstate)
          resetref()
        elif c == keymap['dec_movedist']:
          movedist -= movedistinc
          print(f"move dist now {movedist:0.02f}")
        elif c == keymap['inc_movedist']:
          movedist += movedistinc
          print(f"move dist now {movedist:0.02f}")
        elif c == keymap['dec_turndist']:
          yawdist -= yawdistinc
          print(f"yaw dist now {yawdist:0.02f}")
        elif c == keymap['inc_turndist']:
          yawdist += yawdistinc
          print(f"yaw dist now {yawdist:0.02f}")
        elif c == keymap['ratedown']:
          if args.speedup > 0.1 + 1e-3:
            args.speedup -= 0.1
          print(f"speedup now {args.speedup:0.02f}")
        elif c == keymap['rateup']:
          args.speedup += 0.1
          print(f"speedup now {args.speedup:0.02f}")
        elif c == keymap['help']:
          printhelp()
        else:
          print("Unmapped key", c)

        bodyrot = R.from_euler('ZYX', [controller.ref.yawdes, 0, 0])
        controller.ref.posdes += bodyrot.apply(bodymove)
