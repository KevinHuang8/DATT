import os
import sys
from enum import Enum
from pathlib import Path
import importlib.util


from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC

from DATT.learning.base_env import BaseQuadsimEnv
from DATT.learning.tasks import (hover,
                                    trajectory_fbff, 
                                    trajectory_fbff_vel,
                                    yawflip, 
                                    trajectory_fbff_yaw
                                    )
from DATT.learning.configuration.configuration import RefConfiguration
from DATT.learning.refs import (
        lineref, square_ref, circle_ref, random_zigzag, setpoint_ref, polynomial_ref, random_zigzag_yaw,
        chained_poly_ref, mixed_trajectory_ref, gen_trajectory, pointed_star, closed_polygon)

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

DEFAULT_LOG_DIR = Path(thisdir) / 'logs'
DEFAULT_DATA_DIR = Path(thisdir) / 'data'
CONFIG_DIR = Path(thisdir) / 'configuration'
CONFIG_DIR.mkdir(exist_ok=True)
SAVED_POLICY_DIR = Path(thisdir) / 'saved_policies'
SAVED_POLICY_DIR.mkdir(exist_ok=True)

def import_config(config_filename):
    spec = importlib.util.spec_from_file_location("config", CONFIG_DIR / config_filename)
    
    config_module = importlib.util.module_from_spec(spec)
    sys.modules['config'] = config_module
    spec.loader.exec_module(config_module)
    try:
        return config_module.config
    except AttributeError:
        raise ValueError(f'Config file {config_filename} must define a config object named `config`.')