import sys

import numpy as np

from DATT.quadsim.sim import QuadSim
from DATT.quadsim.models import IdentityModel

from DATT.refs.pointed_star import NPointedStar
from DATT.learning.configs import *

from DATT.controllers  import cntrl_config_presets, ControllersZoo
from DATT.configuration.configuration import AllConfig
from DATT.refs import TrajectoryRef

from DATT.python_utils.plotu import subplot, set_3daxes_equal


import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--cntrl_config', default='datt_hover_config', type=str, 
                        help='Pick or Make a config preset from DATT/quadsim/controllers/cntrl_config_presets')
    parser.add_argument('--cntrl', default=ControllersZoo.DATT, type=ControllersZoo)
    parser.add_argument('--env_config', default='default_hover.py')
    parser.add_argument('-r', '--ref', dest='ref', type=TrajectoryRef, default=TrajectoryRef.LINE_REF)
    parser.add_argument('--seed', type=int, default=0)






    args = parser.parse_args()

    config : AllConfig = import_config(args.env_config)


    posdes = np.array((1.0, 1.0, 1.0))
    # yawdes = np.pi / 2
    yawdes = 0.0
    dt = 0.02
    vis = True
    plot = True

    t_end = 25.0

    # Loading refs
    seed = args.seed
    ref = args.ref.ref(config.ref_config, 
                       seed=seed, 
                       env_diff_seed=config.training_config.env_diff_seed)
    # ref = NPointedStar(n_points=5, speed=2, radius=1)

    # Loading drone configs
    model = IdentityModel()

    # Loading sim
    quadsim = QuadSim(model, vis=vis)

    # Loading controller
    cntrl : ControllersZoo = args.cntrl
    cntrl_config = getattr(cntrl_config_presets, args.cntrl_config, "Config not found")
    controller = cntrl.cntrl(config, {cntrl._value_ : cntrl_config})
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
