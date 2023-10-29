import numpy as np
import yaml
import torch
from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR, TEST_POLICY_DIR
from quadsim.learning.utils.adaptation_network import AdaptationNetwork
from stable_baselines3.common.env_util import make_vec_env

# importing MPPI
from quadrotor_torch import Quadrotor_torch
from param_torch import Param, Timer
import controller_torch
from pathlib import Path
import os
from Controllers.ctrl_config import select_policy_config_

from Opt_Nonlinear_SysID_Quad.controllers import QuadrotorPIDController
from Opt_Nonlinear_SysID_Quad.environments import LearnedKernel, FixedLearnedKernel


class ControllerBackbone():
    def __init__(self, isSim, 
                 policy_config='trajectory', 
                 adaptive=False, 
                 pseudo_adapt=False,
                 adapt_smooth = False,
                 explore_type = 'random',
                 init_run = False):
        
        self.isSim = isSim
        self.mass = 0.04
        self.g = 9.8
        self.e_dims = 0
        self.adaptive = adaptive

        self.pseudo_adapt = pseudo_adapt
        self.adaptation_terms = np.zeros(4)
        self.adaptation_mean = np.zeros((1, 4))
        self.adapt_smooth = adapt_smooth
        self.explore_type = explore_type
        self.exploration_dir = '/home/rwik/proj/Drones/icra_23/Opt_Nonlinear_SysID_Quad/Opt_Nonlinear_SysID_Quad/hessian_bank/traj_ceed_1/1D/seed0/'

        self.init_run = init_run
        # self.I = np.array([[3.144988, 4.753588, 4.640540],
        #                    [4.753588, 3.151127, 4.541223],
        #                    [4.640540, 4.541223, 7.058874]])*1e-5
        self.I = np.array([16.571710e-6, 16.655602e-6, 29.261652e-6])
        
        self.prev_t = None
        self.dt_prev = None
        
        self.dt = None
        self.offset_pos = np.zeros(3)

        self.time_horizon = 10
        self.policy_config = policy_config
        
        # Classical adaptation
        self.v_prev = np.zeros(3)
        self.v_hat = np.zeros(3)
        self.wind_adapt_term = np.zeros(3)
        self.wind_adapt_term_t = np.zeros(3)


        # naive params
        self.lamb = 0.1

        # L1 params
        self.runL1 = True # L1 v/s naive toggle
        self.filter_coeff = 5
        self.A = -0.2
        self.count = 0

    def updateDt(self,t):

        if self.prev_t is None:
            self.dt = 0.02
        else:
            self.dt = t - self.prev_t
        
        if self.dt < 0:
            self.dt = 0

    def select_policy_configs(self,):
        policy_dict = select_policy_config_(self.policy_config)
        self.task = policy_dict["task"]
        self.policy_name = policy_dict["policy_name"]
        self.config_filename = policy_dict["config_filename"]
        self.adaptive_policy_name = policy_dict["adaptive_policy_name"]
        self.body_frame = policy_dict["body_frame"]
        self.relative = policy_dict["relative"]
        self.log_scale = policy_dict["log_scale"]
        self.e_dims = policy_dict["e_dims"]
        self.u_struct = policy_dict["u_struct"]
        self.adaptation_warmup = policy_dict['adaptation_warmup']

    def set_policy(self,):

        self.select_policy_configs()

        self.algo = RLAlgo.PPO
        # self.config = None

        self.algo_class = self.algo.algo_class()

        config = import_config(self.config_filename)
        # self.config = config
        try :
            self.env = self.task.env()(
                config=config,
                log_scale = self.log_scale,
                body_frame = self.body_frame,
                relative = self.relative,
                u_struct = self.u_struct,
            )
        except:
            self.env = self.task.env()(
                config=config,
                log_scale = self.log_scale,
                body_frame = self.body_frame,
            )
        
        self.policy = self.algo_class.load(TEST_POLICY_DIR / f'{self.policy_name}', self.env)
        if self.adaptive == True and self.pseudo_adapt == False:
            self.adaptive_policy = AdaptationNetwork(14, self.e_dims, complex=True)
            self.adaptive_policy.load_state_dict(torch.load(TEST_POLICY_DIR / f'{self.adaptive_policy_name}', map_location='cuda:0'))
        self.prev_pos = 0.
    
    def set_MPPI_controller(self,):
        config_dir = config_dir = os.path.dirname(os.path.abspath(__file__)) + "/mppi_config"

        # config_dir = "/mnt/hdd/drones/crazyswarm/ros_ws/src/crazyswarm/scripts/DroneLearning/Controllers/mppi_config"
        # config_dir = "/home/guan/ya/rwik/drones/crazyswarm/ros_ws/src/crazyswarm/scripts/DroneLearning/Controllers/mppi_config"
        with open(config_dir + "/hover.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.param_MPPI = Param(config, MPPI=True)
        env_MPPI = Quadrotor_torch(self.param_MPPI, config)
        controller = controller_torch.MPPI_thrust_omega(env_MPPI, config)
        
        return controller

    def set_BC_policy(self, ):
        from imitation.algorithms import bc
        BC = 'MPPI_BC'
        # bc_policy_name = BC + '/' 'ppo_mppi_bc'
        # bc_policy_name = BC + '/' 'ppo-mppi_zigzag_bf_rel_bc'
        # bc_policy_name = BC + '/' 'ppo-mppi_zigzag_bc'
        bc_policy_name = BC + '/' 'ppo-mppi_zigzag_xy_bc'
        self.body_frame = False
        self.relative = False
        self.bc_policy = bc.reconstruct_policy(TEST_POLICY_DIR / f'{bc_policy_name}')
    
    def set_PID_torch(self, ):
        from Opt_Nonlinear_SysID_Quad.param_torch import Param as param_explore

        with open('/home/rwik/proj/Drones/icra_23/Opt_Nonlinear_SysID_Quad/Opt_Nonlinear_SysID_Quad/zigzag.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        param = param_explore(config, MPPI=True)
        param.ref_traj_func = None

        param.sim_dt = 0.02
        param.dt = 0.02

        kernel = FixedLearnedKernel()
        # Aker = np.load(self.exploration_dir+'aker.npy')
        aker = torch.tensor(0.01*np.random.randn(3,3), dtype=torch.float32)
        controller = QuadrotorPIDController(torch.tensor([6, 4, 1.5], dtype=torch.float32), param, Adrag=torch.zeros(3,12), optimize_Adrag=True, control_angvel=True, kernel=kernel)
        # controller_params = np.load('/home/rwik/proj/Drones/icra_23/Opt_Nonlinear_SysID_Quad/Opt_Nonlinear_SysID_Quad/data/explore_circle_aggressive_opt_controller.npz', allow_pickle=True)
        # controller_params = np.load('/home/rwik/proj/Drones/icra_23/Opt_Nonlinear_SysID_Quad/Opt_Nonlinear_SysID_Quad/data/random_plate_0_policy_params.npz', allow_pickle=True)
        # gains = torch.tensor([8.9898, 8.2800, 3.7587], dtype=torch.float)
        gains = torch.tensor([ 3.9785,  7.4346, -0.1567])
        aker = torch.tensor([[ 0.0333,  0.0672,  0.0859,  0.0841, -0.1125, -0.3306,  0.1373,  0.0065,
         -0.1780, -0.0017, -0.0149,  0.0704],
        [-0.0758,  0.0009, -0.0571, -0.0469, -0.0630, -0.3979,  0.2740, -0.0286,
         -0.6609,  0.1521,  0.0013, -0.0346],
        [-0.2763, -0.0192, -0.3056, -0.3024, -0.0485, -0.0668,  0.0256,  0.3047,
         -0.0843,  0.0319,  0.1334, -0.3213]])
        controller.update_params([gains, aker])
        return controller

    def _response(self, fl1, response_inputs):
        raise NotImplementedError
    
    def response(self, fl=1, **kwargs):
        return self._response(fl, **kwargs)
    
    # Classical Adaptation techniques
    def naive_adaptation(self, a_t, f_t):
        unity_mass = 1
        g_vec = np.array([0, 0, -1]) * self.g

        adapt_term = unity_mass * a_t - unity_mass * g_vec - f_t

        self.wind_adapt_term = (1 - self.lamb) * self.wind_adapt_term + self.lamb * adapt_term

    def L1_adaptation(self, dt, v_t, f_t):
        unit_mass = 1
        g_vec = np.array([0, 0, -1]) * self.g
        # alpha = np.exp(-dt * self.filter_coeff)
        alpha = 0.99
        # print(alpha)
        phi = 1 / self.A * (np.exp(self.A * dt) - 1)

        a_t_hat = g_vec + f_t / unit_mass - self.wind_adapt_term_t + self.A * (self.v_hat - v_t)
        
        self.v_hat += a_t_hat * dt
        v_tilde = self.v_hat - v_t
        
        self.wind_adapt_term_t = 1 / phi * np.exp(self.A * dt) * v_tilde
        self.wind_adapt_term = -(1 - alpha) * self.wind_adapt_term_t + alpha * self.wind_adapt_term 

