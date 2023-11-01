from DATT.learning.tasks import DroneTask
from DATT.quadsim.controllers.cntrl_config import PIDConfig, MPPIConfig, DATTConfig

# DATT hover
datt_hover_config = DATTConfig()

# Simple DATT w/ feedforward, without any adaptation
datt_config = DATTConfig()
datt_config.policy_name = 'traj_mixed2D_all_refs_diffaxis2_17500000_steps.zip'
datt_config.task = DroneTask.TRAJFBFF
datt_config.config_filename = 'trajectory_latency.py'
datt_config.load_config()

# DATT w/ feedforward and L1 adaptation
datt_adaptive_L1_config = DATTConfig()
datt_adaptive_L1_config.policy_name = 'traj_mixed2D_wind_adaptive2_REAL'
datt_adaptive_L1_config.task = DroneTask.TRAJFBFF
datt_adaptive_L1_config.config_filename = 'trajectory_wind_adaptive.py'
datt_adaptive_L1_config.adaptive = True
datt_adaptive_L1_config.adaptation_type = 'l1'
datt_adaptive_L1_config.adaptive_policy_name = None
datt_adaptive_L1_config.load_config()

# DATT w/ feedforward and RMA adaptation
datt_adaptive_RMA_config = DATTConfig()
datt_adaptive_RMA_config.policy_name = 'traj_mixed2D_wind_adaptive2_REAL'
datt_adaptive_RMA_config.task = DroneTask.TRAJFBFF
datt_adaptive_RMA_config.config_filename = 'trajectory_wind_adaptive.py'
datt_adaptive_RMA_config.adaptive = True
datt_adaptive_RMA_config.adaptation_type = 'rma'
datt_adaptive_RMA_config.adaptive_policy_name = 'wind_adaptation_net_RMA'
datt_adaptive_RMA_config.load_config()


pid_config = PIDConfig()


mppi_config = MPPIConfig()