from DATT.learning.tasks import DroneTask
from DATT.controllers.cntrl_config import PIDConfig, MPPIConfig, DATTConfig

# DATT hover
datt_hover_config = DATTConfig()

# Simple DATT w/ feedforward, without any adaptation
datt_config = DATTConfig()
datt_config.policy_name = 'datt'
datt_config.task = DroneTask.TRAJFBFF


# DATT w/ feedforward and L1 adaptation
datt_adaptive_L1_config = DATTConfig()
datt_adaptive_L1_config.policy_name = 'datt_wind_adaptive'
datt_adaptive_L1_config.task = DroneTask.TRAJFBFF
datt_adaptive_L1_config.adaptive = True
datt_adaptive_L1_config.adaptation_type = 'l1'
datt_adaptive_L1_config.adaptive_policy_name = None

# DATT w/ feedforward and RMA adaptation
datt_adaptive_RMA_config = DATTConfig()
datt_adaptive_RMA_config.policy_name = 'datt_wind_adaptive'
datt_adaptive_RMA_config.task = DroneTask.TRAJFBFF
datt_adaptive_RMA_config.adaptive = True
datt_adaptive_RMA_config.adaptation_type = 'rma'
datt_adaptive_RMA_config.adaptive_policy_name = 'wind_RMA'


pid_config = PIDConfig()


mppi_config = MPPIConfig()