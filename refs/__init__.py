import numpy as np
from pathlib import Path
from enum import Enum

from DATT.configuration.configuration import AllConfig, RefConfiguration
from DATT.refs import (
        lineref, square_ref, circle_ref, random_zigzag, setpoint_ref, polynomial_ref, random_zigzag_yaw,
        chained_poly_ref, mixed_trajectory_ref, gen_trajectory, pointed_star, closed_polygon)

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