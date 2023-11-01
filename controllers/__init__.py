from enum import Enum
from DATT.controllers.pid_controller import PIDController
from DATT.controllers.mppi_controller import MPPIController
from DATT.controllers.datt_controller import DATTController
from DATT.controllers.cntrl_config import *
class ControllersZoo(Enum):
    PID = 'pid'
    MPPI = 'mppi'
    DATT = 'datt'
    
    def cntrl(self, config, cntrl_configs : dict):
        pid_config = cntrl_configs.get('pid', PIDConfig())
        mppi_config = cntrl_configs.get('mppi', MPPIConfig())
        datt_config = cntrl_configs.get('datt', DATTConfig())
        

        return {
            ControllersZoo.PID : PIDController(config, pid_config),
            ControllersZoo.MPPI : MPPIController(config, mppi_config),
            ControllersZoo.DATT : DATTController(config, datt_config)

        }[ControllersZoo(self._value_)]