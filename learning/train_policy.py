import importlib.util
import os
import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path

import torch
import numpy as np
from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from DATT.learning.base_env import BaseQuadsimEnv
from DATT.learning.configuration.configuration import AllConfig, RefConfiguration
from DATT.learning.tasks import (hover,
                                    trajectory_fbff, 
                                    trajectory_fbff_vel,
                                    yawflip, 
                                    trajectory_fbff_yaw
                                    )
from DATT.learning.refs import (
        lineref, square_ref, circle_ref, random_zigzag, setpoint_ref, polynomial_ref, random_zigzag_yaw,
        chained_poly_ref, mixed_trajectory_ref, gen_trajectory, pointed_star, closed_polygon)
from DATT.learning.utils.feedforward_feature_extractor import \
    FeedforwardFeaturesExtractor

class DroneTask(Enum):
    HOVER = 'hover'
    YAWFLIP = 'yawflip'
    TRAJFBFF = 'trajectory_fbff'
    TRAJFBFF_VEL = 'trajectory_fbff_vel'
    TRAJFBFF_YAW = 'trajectory_fbff_yaw'

    def env(self) -> BaseQuadsimEnv:
        return {
            DroneTask.HOVER: hover.HoverEnv,
            DroneTask.YAWFLIP: yawflip.YawflipEnv,
            DroneTask.TRAJFBFF: trajectory_fbff.TrajectoryEnv,
            # V Below: not updated & not working
            DroneTask.TRAJFBFF_VEL: trajectory_fbff_vel.TrajectoryEnv,
            DroneTask.TRAJFBFF_YAW: trajectory_fbff_yaw.TrajectoryYawEnv,
        }[DroneTask(self._value_)]

    def is_trajectory(self):
        return self in [DroneTask.TRAJFBFF, DroneTask.TRAJFBFF_YAW, DroneTask.TRAJFBFF_VEL]

class RLAlgo(Enum):
    PPO = 'ppo'
    A2C = 'a2c'
    DDPG = 'ddpg'
    SAC = 'sac'
    TD3 = 'td3'

    def algo_class(self):
        return {
            RLAlgo.PPO: PPO,
            RLAlgo.A2C: A2C,
            RLAlgo.DDPG: DDPG,
            RLAlgo.SAC: SAC,
            RLAlgo.TD3: TD3,
        }[RLAlgo(self._value_)]
    
class TrajectoryRef(Enum):
    LINE_REF = 'line_ref'
    SQUARE_REF = 'square_ref'
    CIRCLE_REF = 'circle_ref'
    RANDOM_ZIGZAG = 'random_zigzag'
    RANDOM_ZIGZAG_YAW = 'random_zigzag_yaw'
    SETPOINT = 'setpoint'
    POLY_REF = 'poly_ref'
    CHAINED_POLY_REF = 'chained_poly_ref'
    MIXED_REF = 'mixed_ref'
    GEN_TRAJ = 'gen_traj'
    POINTED_STAR = 'pointed_star'
    CLOSED_POLY = 'closed_poly'

    # def ref(self, y_max=0.0, seed=None, init_ref=None, diff_axis=False, z_max=0.0, env_diff_seed=False, include_all=False, ref_name=None, **kwargs):
    def ref(self, config: RefConfiguration, seed=None, env_diff_seed=False, **kwargs):
        if self._value_ == 'gen_traj':
            return gen_trajectory.main_loop(saved_traj=config.ref_name, parent=Path().absolute() / 'refs')
        return {
            TrajectoryRef.LINE_REF: lineref.LineRef(D=1.0, altitude=0.0, period=1),
            TrajectoryRef.SQUARE_REF: square_ref.SquareRef(altitude=0, D1=1.0, D2=0.5, T1=1.0, T2=0.5),
            TrajectoryRef.CIRCLE_REF: circle_ref.CircleRef(altitude=0, rad=0.5, period=2.0),
            TrajectoryRef.RANDOM_ZIGZAG: random_zigzag.RandomZigzag(max_D=np.array([1, config.y_max, config.z_max]), min_dt=0.6, max_dt=1.5, diff_axis=config.diff_axis, env_diff_seed=env_diff_seed, seed=seed, **kwargs),
            TrajectoryRef.RANDOM_ZIGZAG_YAW: random_zigzag_yaw.RandomZigzagYaw(max_D=np.array([1, config.y_max, config.z_max]), min_dt=0.6, max_dt=1.5, seed=seed, **kwargs),
            TrajectoryRef.SETPOINT: setpoint_ref.SetpointRef(setpoint=(0.5, 0.5, 0)),
            TrajectoryRef.POLY_REF: polynomial_ref.PolyRef(altitude=0, use_y=(config.y_max > 0), seed=seed, t_end=10.0, degree=7, env_diff_seed=env_diff_seed, **kwargs),
            TrajectoryRef.CHAINED_POLY_REF: chained_poly_ref.ChainedPolyRef(altitude=0, use_y=(config.y_max > 0), seed=seed, min_dt=1.5, max_dt=4.0, degree=3, env_diff_seed=env_diff_seed, **kwargs),
            TrajectoryRef.MIXED_REF: mixed_trajectory_ref.MixedTrajectoryRef(altitude=0, include_all=config.include_all, init_ref=config.init_ref, ymax=config.y_max, zmax=config.z_max, diff_axis=config.diff_axis, env_diff_seed=env_diff_seed, seed=seed, **kwargs),
            TrajectoryRef.POINTED_STAR: pointed_star.NPointedStar(random=True, seed=seed, env_diff_seed=env_diff_seed, **kwargs),
            TrajectoryRef.CLOSED_POLY: closed_polygon.ClosedPoly(random=True, seed=seed, env_diff_seed=env_diff_seed, **kwargs)
        }[TrajectoryRef(self._value_)]

thisdir = os.path.dirname(os.path.realpath(__file__))

DEFAULT_LOG_DIR = Path().absolute() / 'logs'
DEFAULT_DATA_DIR = Path().absolute() / 'data'
CONFIG_DIR = Path().absolute() / 'configuration'
CONFIG_DIR.mkdir(exist_ok=True)
SAVED_POLICY_DIR = Path(thisdir) / 'saved_policies'
SAVED_POLICY_DIR.mkdir(exist_ok=True)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-t', '--task', dest='task', 
        type=DroneTask, default=DroneTask.HOVER,
        help='Name of the task. Defined in ./tasks/ Default: hover'
    )
    parser.add_argument('-a', '--algo', dest='algo', 
        type=RLAlgo, default=RLAlgo.PPO,
        help='Name of the RL algorithm to train the policy. Default: ppo'
    )
    parser.add_argument('-c', '--config', dest='config',
        default='default_hover.py',
        help='Name of the configuration .py file. Default: default_hover.py'    
    )
    parser.add_argument('-n', '--name', dest='name', 
        default=None,
        help='Name of the policy to train. If such policy already exists in ./saved_policies/, continues training it.'
    )
    parser.add_argument('-d', '--log-dir', dest='log_dir',
        default=None,
        help='The directory to log training info to. Can run tensorboard from this directory to view.'   
    )
    parser.add_argument('-ts', '--timesteps', dest='timesteps',
        type=int, default=1e6,
        help='Number of timesteps to train for. Default: 1 million'    
    )
    parser.add_argument('-sd', '--save-data', dest='save_data',
        type=bool, default=False,
        help='bool, whether to save state transition data for offline supervised learning, etc.'
    )
    parser.add_argument('-dd', '--data-dir', dest='data_dir',
        default=None,
        help='Directory to save data to, if saving data.'
    )
    parser.add_argument('-ch', '--checkpoint', dest='checkpoint',
        type=bool, default=False,
        help='Whether to save checkpoints.'
    )
    parser.add_argument('-de', '--device', dest='device',
        type=int, default=0,
        help='GPU ID to use.'
    )
    
    parser.add_argument('--n-envs', type=int, help='How many "parallel" environments to run', default=10)
    parser.add_argument('-r', '--ref', dest='ref', type=TrajectoryRef, default=TrajectoryRef.LINE_REF)
    parser.add_argument('--seed', dest='seed', type=int, default=None,
        help='Seed to use for randomizing reference trajectories during training.'
    )

    args = parser.parse_args()

    return args

def find_default_name_num(dir, prefix):
    seen_nums = set()
    for name in os.listdir(dir):
        if name.startswith(f'{prefix}_'):
            try:
                num = int(name[len(prefix) + 1:])
            except ValueError:
                pass
            else:
                seen_nums.add(num)
    
    num = 0
    while num in seen_nums:
        num += 1

    return f'{prefix}_{num}'


def import_config(config_filename):
    spec = importlib.util.spec_from_file_location("config", CONFIG_DIR / config_filename)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules['config'] = config_module
    spec.loader.exec_module(config_module)
    try:
        return config_module.config
    except AttributeError:
        raise ValueError(f'Config file {config_filename} must define a config object named `config`.')

def train():
    args = parse_args()
    
    task: DroneTask = args.task
    policy_name = args.name
    ref = args.ref
    log_dir = args.log_dir
    ts = args.timesteps
    algo = args.algo
    config_filename = args.config
    save_data = args.save_data
    checkpoint = args.checkpoint
    data_dir = args.data_dir
    device = args.device
    seed = args.seed
    n_envs = args.n_envs

    if policy_name is None:
        policy_name = f'{task.value}_{algo.value}_policy'

    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR / f'{policy_name}_logs'
        log_dir.mkdir(exist_ok=True, parents=True)
    if not log_dir.exists():
        raise FileNotFoundError(f'{log_dir} does not exist')

    if not (CONFIG_DIR / config_filename).exists():
        raise FileNotFoundError(f'{config_filename} is not a valid config file')

    if save_data and data_dir is None:
        data_dir = DEFAULT_DATA_DIR / policy_name
        data_dir.mkdir(exist_ok=True, parents=True)
    if save_data:
        if not data_dir.exists():
            raise FileNotFoundError(f'{data_dir} does not exist')
        data_filename = data_dir / find_default_name_num(data_dir, f'{policy_name}_data')
        print(f'Saving data to {data_filename}')
        with open(data_filename, 'w+') as f:
            pass
    else:
        data_filename = None

    config: AllConfig = import_config(config_filename)
    env_kwargs={
        'config': config,
        'save_data': save_data, 
        'data_file': data_filename,
    }

    if task.is_trajectory():
        env_kwargs['ref'] = ref
        if seed is not None:
            env_kwargs['seed'] = seed
        else:
            env_kwargs['seed'] = np.random.randint(0, 100000)

    env_class = task.env()

    if issubclass(env_class, VecEnv):
      env = VecMonitor(env_class(n_envs))
    else:
      env = make_vec_env(env_class, n_envs=n_envs, env_kwargs=env_kwargs)

    algo_class = algo.algo_class()

    if not (SAVED_POLICY_DIR / f'{policy_name}.zip').exists():
        features_extractor_kwargs = {}        
        if task.is_trajectory():
            if config.policy_config.conv_extractor:
                features_extractor_class = FeedforwardFeaturesExtractor
            else:
                features_extractor_class = FlattenExtractor

            features_extractor_kwargs['extra_state_features'] = 0
            extra_dims = task.env()(config=config).extra_dims
            features_extractor_kwargs['extra_state_features'] += extra_dims
            if task == DroneTask.TRAJFBFF or task == DroneTask.TRAJFBFF_VEL or task == DroneTask.TRAJ_POLY:
                features_extractor_kwargs['extra_state_features'] += 3
            elif task == DroneTask.TRAJFBFF_YAW:
                features_extractor_kwargs['extra_state_features'] += 4

            if task == DroneTask.TRAJFBFF and not config.policy_config.fb_term:
                features_extractor_kwargs['extra_state_features'] -= 3

            if task == DroneTask.TRAJFBFF_VEL:
                features_extractor_kwargs['dims'] = 6
            elif task == DroneTask.TRAJFBFF_YAW:
                features_extractor_kwargs['dims'] = 4
        else:
            features_extractor_class = FlattenExtractor

        print(f'Using feature extractor: {features_extractor_class}')
        net_arch = [dict(pi=[64, 64, 64], vf=[64, 64, 64])]
        if issubclass(algo_class, OffPolicyAlgorithm):
            net_arch = [64, 64, 64]

        policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=net_arch,
                     features_extractor_class=features_extractor_class,
                     features_extractor_kwargs=features_extractor_kwargs
        )

        kwargs = {}
        if issubclass(algo_class, OffPolicyAlgorithm):
            kwargs['train_freq'] = (5000, 'step')

        policy_network_type = 'MlpPolicy'
        print(f'Using policy network type: {policy_network_type}')
        
        policy: BaseAlgorithm = algo_class(
            policy_network_type, 
            env, 
            tensorboard_log=log_dir, 
            policy_kwargs=policy_kwargs,
            device=f'cuda:{device}',
            verbose=0,
            **kwargs
        )
    else:
        policy: BaseAlgorithm = algo_class.load(SAVED_POLICY_DIR / f'{policy_name}.zip', env)
        print('CONTINUING TRAINING!')

    if checkpoint:
        checkpoint_callback = CheckpointCallback(
            save_freq=250000,
            save_path=SAVED_POLICY_DIR,
            name_prefix=policy_name
        )
    else:
        checkpoint_callback = None

    policy.learn(total_timesteps=ts, progress_bar=True, callback=checkpoint_callback)
    policy.save(SAVED_POLICY_DIR / policy_name)

if __name__ == "__main__":
    train()
