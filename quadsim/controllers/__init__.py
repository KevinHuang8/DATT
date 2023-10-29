# from Controllers.hover_ppo_controller import *
# from Controllers.bc_controller import BC_Controller
# from Controllers.traj_ppo_controller import PPOController_trajectory
# from Controllers.traj_ppo_controller_adapt import PPOController_trajectory_adaptive
# from Controllers.traj_ppo_controller_ustruct import PPOController_trajectory_ustruct
# from Controllers.traj_ppo_controller_adapt_L1 import PPOController_trajectory_L1_adaptive

# from DATT.quadsim.controllers.pid_controller import PIDController
# from DATT.quadsim.controllers.mppi_controller import MPPIController
# from enum import Enum


# from quadsim_vision.configuration import PIDConfig, MPPIConfig
# class ControllersZoo(Enum):
#     PID = 'pid'
#     MPPI = 'mppi'

#     def cntrl(self, env_config, drone_config, configs : dict):
#         pid_config = configs.get('pid_config', PIDConfig())
#         mppi_config = configs.get('mppi_config', MPPIConfig())

#         return {
#             ControllersZoo.PID : PIDController(env_config, drone_config, pid_config),
#             ControllersZoo.MPPI : MPPIController(env_config, drone_config, mppi_config),
#         }[ControllersZoo(self._value_)]
