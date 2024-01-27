import time
import numpy as np
import matplotlib.pyplot as plt

from os.path import exists
from argparse import ArgumentParser

# from DATT.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR, TrajectoryRef
from DATT.learning.configs import *
from DATT.learning.tasks import DroneTask
from DATT.refs import TrajectoryRef

from DATT.quadsim.visualizer import Vis

from scipy.spatial.transform import Rotation as R


from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from DATT.configuration.configuration import AllConfig

from DATT.learning.adaptation_module import Adapation


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-t', '--task', dest='task',
        type=DroneTask, default=DroneTask.HOVER
        )
    parser.add_argument('-n', '--name', dest='name',
        default=None)
    parser.add_argument('-a', '--algo', dest='algo',
        type=RLAlgo, default=RLAlgo.PPO)
    parser.add_argument('-s', '--steps', dest='eval_steps',
        type=int, default=1000)
    parser.add_argument('-c', '--config', dest='config',
        default='default_hover.py',
        help='Name of the configuration .py file. Default: default_hover.py'
    )
    parser.add_argument('-v', '--viz', dest='viz',
        type=bool, default=False,
        help='Whether to ')
    parser.add_argument('-r', '--rate', dest='rate',
        default=100, type=float,
        help='Rate (Hz) of visualization. Sleeps for 1 / rate between states. Set to negative for uncapped.'
    )
    parser.add_argument('--ref', dest='ref',
        default=TrajectoryRef.LINE_REF, type=TrajectoryRef,
        help='Rate (Hz) of visualization. Sleeps for 1 / rate between states. Set to negative for uncapped.'
    )
    parser.add_argument('--seed', dest='seed', type=int, default=None)
    args = parser.parse_args()

    return args


def eval():
    args = parse_args()

    task: DroneTask = args.task
    policy_name = args.name
    algo = args.algo
    eval_steps = args.eval_steps
    config_filename = args.config
    viz = args.viz
    ref = args.ref
    seed = args.seed

    if not exists(SAVED_POLICY_DIR / f'{policy_name}.zip'):
        raise ValueError(f'policy not found: {policy_name}')
    if not exists(CONFIG_DIR / config_filename):
        raise FileNotFoundError(f'{config_filename} is not a valid config file')

    algo_class = algo.algo_class()

    config: AllConfig = import_config(config_filename)

    l1_sim = config.sim_config.L1_simulation

    if task.is_trajectory():
        if seed is None:
            seed = np.random.randint(0, 100000)
            fixed_seed = False
        else:
            fixed_seed = True
        print(seed)
        evalenv = task.env()(config=config, ref=ref, seed=seed, fixed_seed=fixed_seed)
    else:
        evalenv = task.env()(config=config)

    policy = algo_class.load(SAVED_POLICY_DIR / f'{policy_name}.zip')

    control_error_avg = 0
    adaptation_module = Adapation()
    count = 0
    if viz:
        vis = Vis() 
    try:
        while True:
            count += 1
            total_r = 0
            obs = evalenv.reset()
            adaptation_module.reset()
            all_states = []
            all_rewards = []
            all_actions = []
            all_ang_vel_actual = []
            all_ang_vel_desired = []
            des_traj = []
            control_errors = []
            all_wind = []
            l1_terms = []
            try:
                print('wind field', evalenv.wind_vector)
                print('mass', evalenv.model.mass)
                print('k', evalenv.k)
            except:
                pass
            for i in range(eval_steps):
                action, _states = policy.predict(obs, deterministic=True)

                act = action
                obs, rewards, dones, info = evalenv.step(act)

                state = evalenv.getstate()

                wind_vector = evalenv.wind_vector
                all_wind.append(wind_vector.copy())
                if l1_sim:
                    l1_terms.append(evalenv.adaptation_module.d_hat.copy())
                all_actions.append(action)
                all_rewards.append(rewards)
                all_states.append(np.r_[state.pos, state.vel, obs[6:10]])
                # all_pid_actions.append(pid_action[0])
                # all_bc_actions.append(bc_action)
                # if not isinstance(evalenv, VecEnv):
                #   all_ang_vel_actual.append(info['motor'][3])
                #   all_ang_vel_desired.append(info['motor'][1])

                try:
                    des_traj.append(evalenv.ref.pos(evalenv.t))
                except:
                    pass

                total_r += rewards

                control_error = np.linalg.norm(state.pos - evalenv.ref.pos(evalenv.t))
                control_errors.append(control_error)

                if viz:
                    vis.set_state(state.pos.copy(), state.rot)
                if args.rate > 0:
                    time.sleep(1.0 / args.rate)

            all_wind = np.array(all_wind)
            l1_terms = np.array(l1_terms)
            all_states = np.array(all_states)
            all_actions = np.array(all_actions)
            all_rewards = np.array(all_rewards)
            all_ang_vel_desired = np.array(all_ang_vel_desired)
            all_ang_vel_actual =  np.array(all_ang_vel_actual)
            des_traj = np.array(des_traj)
            
            if viz:
                plt.figure()
                ax = plt.subplot(3, 1, 1)
                plt.plot(range(eval_steps), all_states[:, 0])
                if des_traj.size > 0:
                    plt.plot(range(eval_steps), des_traj[:, 0])
                plt.subplot(3, 1, 2)
                plt.plot(range(eval_steps), all_states[:, 1])
                if des_traj.size > 0:
                    plt.plot(range(eval_steps), des_traj[:, 1])
                plt.subplot(3, 1, 3)
                plt.plot(range(eval_steps), all_states[:, 2])
                if des_traj.size > 0:
                    plt.plot(range(eval_steps), des_traj[:, 2])
                plt.suptitle('PPO (sim) des vs. actual pos mass : {}'.format(evalenv.model.mass))
                # plt.suptitle('Positions')

                plt.figure()
                ax = plt.subplot(3, 1, 1)
                plt.plot(range(eval_steps), all_wind[:, 0], label='x')
                if l1_sim:
                    plt.plot(range(eval_steps), l1_terms[:, 0], label='L1 x')
                plt.subplot(3, 1, 2, sharex=ax)
                plt.plot(range(eval_steps), all_wind[:, 1], label='y')
                if l1_sim:
                    plt.plot(range(eval_steps), l1_terms[:, 1], label='L1 y')
                plt.subplot(3, 1, 3, sharex=ax)
                plt.plot(range(eval_steps), all_wind[:, 2], label='z')
                if l1_sim:
                    plt.plot(range(eval_steps), l1_terms[:, 2], label='L1 z')
                plt.suptitle('L1 vs. Wind')


                try:
                    plt.figure()
                    plt.plot(all_states[:, 0], all_states[:, 1], label='actual')
                    plt.plot(des_traj[:, 0], des_traj[:, 1], label='desired')
                    plt.legend()
                except:
                    pass

                eulers = np.array([R.from_quat(rot).as_euler('ZYX')[::-1] for rot in all_states[:, 6:10]])
                # plt.figure()
                # ax = plt.subplot(3, 1, 1)
                # plt.plot(range(eval_steps), all_actions[:, 1])
                # plt.plot(range(eval_steps), all_pid_actions[:, 1])
                # # plt.plot(range(eval_steps), all_bc_actions[:, 1])
                # plt.subplot(3, 1, 2, sharex=ax)
                # plt.plot(range(eval_steps), all_actions[:, 2])
                # plt.plot(range(eval_steps), all_pid_actions[:, 2])
                # # plt.plot(range(eval_steps), all_bc_actions[:, 2])
                # plt.subplot(3, 1, 3, sharex=ax)
                # plt.plot(range(eval_steps), all_actions[:, 3], label='PPO')
                # plt.plot(range(eval_steps), all_pid_actions[:, 3], label='PID')
                # # plt.plot(range(eval_steps), all_bc_actions[:, 3], label='BC')
                # plt.legend()
                # plt.suptitle('Ang vel cmds (rad/s)')

                # fig = plt.figure(num="Trajectory")
                # ax = fig.add_subplot(111, projection='3d')
                # plt.plot(all_states[:, 0], all_states[:, 1], all_states[:, 2])
                # plt.xlabel("X (m)")
                # plt.ylabel("Y (m)")
                # ax.set_zlabel("Z (m)")
                # plt.title("Trajectory")
                # set_3daxes_equal(ax)

                # plt.figure()
                # ax1 = plt.subplot(3, 1, 1)
                # plt.plot(range(eval_steps), all_ang_vel_desired[:, 0])
                # plt.plot(range(eval_steps), all_ang_vel_actual[:, 0])
                # plt.subplot(3, 1, 2, sharex=ax1)
                # plt.plot(range(eval_steps), all_ang_vel_desired[:, 1])
                # plt.plot(range(eval_steps), all_ang_vel_actual[:, 1])
                # plt.subplot(3, 1, 3, sharex=ax1)
                # plt.plot(range(eval_steps), all_ang_vel_desired[:, 2], label='des')
                # plt.plot(range(eval_steps), all_ang_vel_actual[:, 2], label='actual')
                # plt.legend()
                # plt.suptitle('Ang vel cmds vs actual (rad/s)')

                # plt.figure()
                # plt.plot(range(eval_steps), all_actions[:, 0], label='PPO')
                # plt.plot(range(eval_steps), all_pid_actions[:, 0], label='PID')
                # plt.plot(range(eval_steps), all_bc_actions[:, 0], label='BC')
                # plt.legend()
                # plt.title('Body z force cmd')

                # subplot(range(eval_steps), all_actions[:, 0], yname='z thrust', title="Body z force cmd")
                # subplot(range(eval_steps), all_pid_actions[:, 0], yname='z thrust PID', title="Body z force cmd")
                # subplot(range(eval_steps), all_actions[:, 1:], yname='ang velocity (rad/s)', title="Angular Velocity cmds")
                # subplot(range(eval_steps), all_actions[:, 1:], yname='ang velocity (rad/s)', title="Angular Velocity cmds PID")
                # subplot(range(eval_steps), eulers, yname="Euler (rad)", title="ZYX Euler Angles")
                # subplot(range(eval_steps), all_rewards, title="Reward per step")
                # plt.savefig("rwik_{}_pos.png".format(str(count).zfill(3)))
                plt.show()
            print(total_r)
            control_error_avg += np.mean(control_errors)
            print('control error', np.mean(control_errors))
    except KeyboardInterrupt:
        print('Control Error: ', control_error_avg / count)

if __name__ == "__main__":
    eval()
