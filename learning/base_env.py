import numpy as np

from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

from scipy.spatial.transform import Rotation as R
from gym import Env, spaces

from DATT.quadsim.sim import QuadSim
from DATT.quadsim.models import IdentityModel, RBModel
from DATT.quadsim.rigid_body import State_struct
from DATT.quadsim.dist import WindField, ConstantForce
from DATT.configuration.configuration import AllConfig
from DATT.configuration.configuration import EnvCondition

from DATT.learning.adaptation_module import Adapation

@dataclass
class ObsData:
    prev_obs: np.ndarray
    action: np.ndarray
    next_obs: np.ndarray
    rew: np.ndarray
    done: bool
    info: dict
    
DATA_EPISODE_SEP = '---'
DATA_SEP = '|'

def save(step_func):
    def step_and_save(*args, **kwargs):
        self: BaseQuadsimEnv = args[0]
        action = args[1]
        if self.save_data:
            curr_obs = self.obs(self.getstate())
            new_obs, rew, done, info = step_func(*args, **kwargs)
            self.data_store.append(ObsData(curr_obs, action, new_obs, rew, done, info))
            self.steps_since_save += 1
            if self.steps_since_save == self.save_interval:
                self.steps_since_save = 0
                # print('Writing data to file...')
                self.save_info()
        else:
            return step_func(*args, **kwargs)
        return new_obs, rew, done, info
    return step_and_save

class BaseQuadsimEnv(Env):
    def __init__(self, 
            config: AllConfig, 
            eval: bool = False, 
            save_data: bool = False, 
            data_file: Optional[Path] = None, 
            save_interval: int = 100000, 
            save_verbose: bool = False
        ):
        """
        config: config object
        eval: If True, do not include env conditions like wind, etc. as part the obs.
        save_data: Whether to record training data (action, obs pairs) to a file
        data_file: if save_data is True, specifies the path of the data file
        save_interval: write to file every `save_interval` training steps
        save_verbose: save all training data, including rew, done, info
        """
        self.config = config
        self.save_data = save_data
        self.data_file = data_file
        self.eval = eval
        self.save_interval = save_interval
        self.steps_since_save = 0
        self.save_verbose = save_verbose
        self.body_frame = self.config.training_config.body_frame
        self.second_order_delay = self.config.sim_config.second_order_delay
        self.L1_simulation = self.config.sim_config.L1_simulation
        
        self.data_store: List[ObsData] = [] 

        # pos (x3), vel (x3), rot (x4, quarternion), ang vel (x3) + env conditions
        self.adapt_config = self.config.adapt_config
        if self.adapt_config is None or self.eval:
            self.included_params: List[EnvCondition] = []
        else:
            self.included_params: List[EnvCondition] = self.adapt_config.include

        if self.save_data and self.data_file is None:
            raise ValueError('Must specify a data file.')
        
        self.reset()

        self.dt = self.config.sim_config.dt()

        extra_dims = 0
        all_mins = np.array([])
        all_maxes = np.array([])
        for condition in self.included_params:
            attr_, dim, mins, maxes = condition.get_attribute(self)
            if isinstance(mins, (float, int)):
                mins = np.ones(dim) * mins
            if isinstance(maxes, (float, int)):
                maxes = np.ones(dim) * maxes
            all_mins = np.r_[all_mins, mins]
            all_maxes = np.r_[all_maxes, maxes]
            extra_dims += dim

        self.base_dims = 10
        obs_dims = self.base_dims + extra_dims
        self.extra_dims = extra_dims
        self.obs_dims = obs_dims
        self.all_mins = np.r_[np.ones(10) * -50, all_mins]
        self.all_maxes = np.r_[np.ones(10) * 50, all_maxes]
        self.observation_shape = (obs_dims,)
        self.observation_space = spaces.Box(low=self.all_mins, high=self.all_maxes)

        self.action_shape = (4,)
        self.action_space = spaces.Box(low=-20 * np.ones(4), high=20 * np.ones(4))
        self.t_end = self.dt * 500

    def save_info(self, new_episode=False):
        with open(self.data_file, 'a') as file:
            for d in self.data_store:
                a2s = lambda a: np.array2string(a, max_line_width=np.inf, separator=',')
                if self.save_verbose:
                    line = f'{a2s(d.prev_obs[:self.obs_dims])}{DATA_SEP}{a2s(d.action)}{DATA_SEP}{a2s(d.next_obs[:self.obs_dims])}{DATA_SEP}{d.rew}{DATA_SEP}{d.done}\n'
                else:
                    line = f'{a2s(d.prev_obs[:self.obs_dims])}{DATA_SEP}{a2s(d.action)}\n'
                file.write(line)
            if new_episode:
                file.write(f'{DATA_EPISODE_SEP}\n')
        self.data_store = []

    def reset(self, state: Optional[State_struct]=None):
        """
        Reset the environment (at given initial state if provided). 
        Randomizes environmental conditions if specified by config.
        """
        if self.save_data:
            self.save_info(new_episode=True)

        drone_config = self.config.drone_config
        sim_config = self.config.sim_config
        self.model = RBModel(
            mass=drone_config.sampler.sample_param(drone_config.mass), 
            I=np.eye(3)*drone_config.sampler.sample_param(drone_config.I), 
            g=drone_config.sampler.sample_param(sim_config.g)
        )
        self.g = drone_config.sampler.sample_param(sim_config.g)
        self.quadsim = QuadSim(self.model, vis=False)

        self.t = 0.0 

        if state is None:
            init_config = self.config.init_config
            state = State_struct(
                        pos=init_config.sampler.sample_param(init_config.pos),
                        vel=init_config.sampler.sample_param(init_config.vel),
                        rot=R.from_euler('ZYX', init_config.sampler.sample_param(init_config.rot), degrees=True),
                        ang=init_config.sampler.sample_param(init_config.ang)
                    )

        wind_config = self.config.wind_config
        self.wind_vector = wind_config.sampler.sample_param(wind_config.dir) 
        if wind_config.is_wind:
            self.dists = [
                ConstantForce(
                    scale=self.wind_vector
                )
            ]
        else:
            self.dists = []

        sim_config = self.config.sim_config
        self.linear_var = sim_config.sampler.sample_param(sim_config.linear_var)
        self.angular_var = sim_config.sampler.sample_param(sim_config.angular_var)
        self.obs_noise = sim_config.sampler.sample_param(sim_config.obs_noise)
        self.latency = sim_config.sampler.sample_param(sim_config.latency)
        self.k = sim_config.sampler.sample_param(sim_config.k)
        self.kw = sim_config.sampler.sample_param(sim_config.kw)
        self.kt = sim_config.sampler.sample_param(sim_config.kt)

        self.quadsim.setstate(state)

        obs = self.obs(state)

        if self.L1_simulation:
            self.prev_thrust_cmd = 0.0
            self.adaptation_module = Adapation()
            # hack, only valid when d is only env param
            obs[10:13] = self.adaptation_module.d_hat
        return obs        

    def getstate(self) -> State_struct:
        """
        Returns the true state of the drone.
        """
        return self.quadsim.getstate()
    
    def obs(self, state: State_struct) -> np.ndarray:
        """
        Returns a noisy observation of the state, augmented with additional
        ground truth environmental conditions, as specified by the adaptation
        configuration.
        """
        noisypos = state.pos + np.random.normal(loc=0.0, scale=self.obs_noise, size=3)
        if self.body_frame:
            vel = state.vel
            rot = state.rot

            noisypos = rot.inv().apply(noisypos)
            vel = rot.inv().apply(vel)

            q = rot.as_quat()

            obs = np.hstack((noisypos, vel, q))
        else:
            obs = np.hstack((noisypos, state.vel, state.rot.as_quat()))

        # Append any additional ground truth environmental conditions to the state
        for condition in self.included_params:
            attr, _, _, _ = condition.get_attribute(self)
            obs = np.hstack((obs, attr))
        return obs

    @save
    def step(self, action):
        u, angvel = action[0], action[1:4]
        u += self.g

        # Time-dependent wind (brownian motion)
        wind_config = self.config.wind_config
        if wind_config.is_wind and wind_config.random_walk:
            self.wind_vector += np.random.normal(loc=0, scale=wind_config.random_walk * self.dt, size=(3,))
            self.dists = [
                ConstantForce(
                    scale=self.wind_vector
                )
            ]

        state = self.quadsim.step_angvel_raw(self.dt, u, angvel,
                dists=self.dists, linear_var=self.linear_var, angular_var=self.angular_var,
                latency=self.latency,
                k=self.k,
                second_order=self.second_order_delay,
                kw=self.kw if self.second_order_delay else 1.0,
                kt=self.kt if self.second_order_delay else 1.0
            )
        self.t += self.dt

        failed = False

        o = self.obs(state)
        # Include L1 simulation; hack, only works when d is 3-dim force pert. and is appended right after the state
        if self.L1_simulation:
            state = self.getstate()
            rot = state.rot
            f = rot.apply(np.array([0, 0, self.prev_thrust_cmd]))
            self.adaptation_module.adaptation_step(state.vel, f)
            o[10:13] = self.adaptation_module.d_hat
        r = self.reward(state, action)
        done = self.t >= self.t_end or failed
        info = dict()

        if failed:
            r -= 10000

        if self.L1_simulation:
            self.prev_thrust_cmd = u

        return o, r, done, info
