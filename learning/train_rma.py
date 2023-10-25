import torch
import os
import numpy as np
import torch.nn.functional as F

from os.path import exists
from argparse import ArgumentParser
from quadsim.learning.train_policy import TrajectoryRef
from tqdm.rich import tqdm, Progress, RateColumn
from rich.progress import MofNCompleteColumn, TimeElapsedColumn

from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR, DEFAULT_LOG_DIR
from quadsim.learning.configuration.configuration import AllConfig
from quadsim.learning.utils.adaptation_network import AdaptationNetwork
from quadsim.learning.base_env import BaseQuadsimEnv

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-t', '--task', dest='task',
        type=DroneTask, default=DroneTask.HOVER
    )
    parser.add_argument('-n', '--name', dest='name',
        default=None)
    parser.add_argument('-an', '--adapt-name', dest='adapt_name',
        default=None,
        help='(optional) Filename of the adaptation network, if different from the policy name.'
    )
    parser.add_argument('-a', '--algo', dest='algo',
        type=RLAlgo, default=RLAlgo.PPO)
    parser.add_argument('-i', '--iterations', dest='train_iterations',
        type=int, default=5000)
    parser.add_argument('-c', '--config', dest='config',
        default='default_hover.py',
        help='Name of the configuration .py file. Default: default_hover.py'
    )
    parser.add_argument('-de', '--device', dest='device',
        type=int, default=0,
        help='GPU ID to use.'
    )


    parser.add_argument('--ref', dest='ref',
        default=TrajectoryRef.LINE_REF, type=TrajectoryRef,
        help='Rate (Hz) of visualization. Sleeps for 1 / rate between states. Set to negative for uncapped.'
    )
    parser.add_argument('--n-envs', dest='n_envs', type=int, default=10)
    parser.add_argument('--subprocess', dest='subprocess', type=bool, default=False)
    parser.add_argument('--ymax', dest='ymax', type=float, default=0.0)
    parser.add_argument('--zmax', dest='zmax', type=float, default=0.0)
    parser.add_argument('--diff-axis', dest='diff_axis', type=bool, default=False)
    parser.add_argument('--relative', dest='relative', type=bool, default=False)
    parser.add_argument('--seed', dest='seed', type=int, default=None)
    parser.add_argument('--body-frame', dest='body_frame', type=bool, default=False)
    parser.add_argument('--log-scale', dest='log_scale', type=bool, default=False)
    parser.add_argument('--second-order', dest='second_order', type=bool, default=False)


    args = parser.parse_args()

    return args

def add_e_dim(fbff_obs: np.ndarray, e: np.ndarray, base_dims=10):
    # fbff_obs shape = (n_envs, fbff_dims)
    
    obs = np.concatenate((fbff_obs[:, :base_dims], e, fbff_obs[:, base_dims:]), axis=1)
    return obs

def remove_e_dim(output_obs: np.ndarray, e_dims: int, base_dims=10, include_extra=False):
    if include_extra:
        obs = np.concatenate([output_obs[:, :base_dims], output_obs[:, (base_dims + e_dims):]], axis=1)
    else:
        obs = output_obs[:, :base_dims]
    return obs

def rollout_adaptive_policy(rollout_len, adaptation_network, policy, evalenv, n_envs, time_horizon, base_dims, e_dims, device, progress=None):
    action_dims = 4

    history = torch.zeros((n_envs, base_dims + action_dims, time_horizon)).to(device)
    all_e_pred = None
    all_e_gt = None
    fbff_obs = remove_e_dim(evalenv.reset(), e_dims, include_extra=True)
    for i in range(rollout_len):
        # shape (n_envs, e_dim)
        e_pred = adaptation_network(history)

        input_obs = add_e_dim(fbff_obs, e_pred.detach().cpu().numpy(), base_dims)

        actions, _states = policy.predict(input_obs, deterministic=True)

        # this obs contains e, which should be removed
        obs, rewards, dones, info = evalenv.step(actions)

        e_gt = obs[:, base_dims:(base_dims + e_dims)]
        e_gt = torch.from_numpy(e_gt).to(device).float()

        if all_e_pred is None:
            all_e_pred = e_pred
            all_e_gt = e_gt
        else:
            all_e_pred = torch.cat((all_e_pred, e_pred), dim=0) 
            all_e_gt = torch.cat((all_e_gt, e_gt), dim=0)


        # just the pos, vel, orientation part of state should be used for prediction of e
        base_states = remove_e_dim(obs, e_dims)
        adaptation_input = np.concatenate((base_states, actions), axis=1)

        adaptation_input = torch.from_numpy(adaptation_input).to(device).float()

        # shift history forward in time
        history = torch.cat((torch.unsqueeze(adaptation_input, -1), history[:, :, :-1].clone()), dim=2)
        
        fbff_obs = remove_e_dim(obs, e_dims, include_extra=True)

        if progress is not None:
            progress[0].update(task_id=progress[1], completed=i + 1)

    return all_e_pred, all_e_gt

def RMA():
    args = parse_args()

    task: DroneTask = args.task
    task_train = task
    policy_name = args.name
    algo = args.algo
    train_iterations = args.train_iterations
    config_filename = args.config
    adapt_name = args.adapt_name

    ref = args.ref
    ymax = args.ymax
    zmax = args.zmax
    seed = args.seed
    relative = args.relative
    body_frame = args.body_frame
    subprocess = args.subprocess
    log_scale = args.log_scale
    n_envs=args.n_envs
    de = args.device
    second_order = args.second_order

    diff_axis = args.diff_axis

    if not exists(SAVED_POLICY_DIR / f'{policy_name}.zip'):
        raise ValueError(f'policy not found: {policy_name}')
    if not exists(CONFIG_DIR / config_filename):
        raise FileNotFoundError(f'{config_filename} is not a valid config file')

    algo_class = algo.algo_class()

    config: AllConfig = import_config(config_filename)
    adapt_config = config.adapt_config
    env_params = adapt_config.include
    time_horizon = adapt_config.time_horizon

    dummy_env = BaseQuadsimEnv(config)
    e_dims = 0
    for param in env_params:
        _, dims, _, _ = param.get_attribute(dummy_env)
        e_dims += dims

    trainenv = task_train.env()(config=config)
    vec_env_class = SubprocVecEnv if subprocess else DummyVecEnv
    evalenv = make_vec_env(task.env(), n_envs=n_envs, vec_env_cls=vec_env_class,
        env_kwargs={
            'config': config,
            'log_scale': log_scale,
            'ref': ref,
            'y_max': ymax,
            'z_max': zmax,
            'diff_axis': diff_axis,
            'relative': relative,
            'body_frame': body_frame,
            'second_order_delay': second_order
        }
    )    

    device = torch.device(f"cuda:{de}" if torch.cuda.is_available() else "cpu")

    policy = algo_class.load(SAVED_POLICY_DIR / f'{policy_name}.zip')

    action_dims = 4
    if adapt_name is not None and os.path.exists(SAVED_POLICY_DIR / f'{policy_name}_adapt' / f'{adapt_name}'):
        adaptation_network_state_dict = torch.load(SAVED_POLICY_DIR / f'{policy_name}_adapt' / f'{adapt_name}', map_location=torch.device('cpu'))
    else:
        adaptation_network_state_dict = None

    if adapt_name is None:
        adapt_name = f'{policy_name}_adapt'

    if not os.path.isdir(SAVED_POLICY_DIR / f'{policy_name}_adapt'):
        os.mkdir(SAVED_POLICY_DIR / f'{policy_name}_adapt')

    adaptation_network = AdaptationNetwork(input_dims=trainenv.base_dims + action_dims, e_dims=e_dims)
    adaptation_network = adaptation_network.to(device)
    if not adaptation_network_state_dict is None:
        adaptation_network.load_state_dict(adaptation_network_state_dict)


    optimizer = torch.optim.Adam(adaptation_network.parameters(), lr=0.001)

    writer = SummaryWriter(DEFAULT_LOG_DIR / f'{adapt_name}_logs')
    running_loss = 0.0
    with Progress(
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        RateColumn(unit='Iterations')
    ) as pg:
        iter = pg.add_task('Iteration', total=train_iterations)
        rollouts = pg.add_task('Rollouts', total=500)

        for i in range(train_iterations):
            optimizer.zero_grad()

            all_e_pred, all_e_gt = rollout_adaptive_policy(500, adaptation_network, policy, evalenv, n_envs, time_horizon, trainenv.base_dims, e_dims, device, progress=(pg, rollouts))

            loss = F.mse_loss(all_e_pred, all_e_gt)
            loss.backward()

            print(f'loss: {loss.detach().cpu().item()}')

            optimizer.step()

            running_loss += loss.detach().cpu().item()

            # Save adaptation network every 50 iterations
            if i % 500 == 0:
                torch.save(adaptation_network.state_dict(), SAVED_POLICY_DIR / f'{policy_name}_adapt' / f'{adapt_name}_{i}')
            if i % 10 == 0:
                writer.add_scalar('training loss', running_loss / 10, i * 500 * n_envs)
                running_loss = 0.0

            pg.update(task_id=iter, completed=i + 1)

        torch.save(adaptation_network.state_dict(), SAVED_POLICY_DIR / f'{policy_name}_adapt' / f'{adapt_name}')

    
if __name__ == "__main__":
    RMA()
