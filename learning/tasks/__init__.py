from enum import Enum
from DATT.learning.tasks import (hover,
                                    trajectory_fbff, 
                                    trajectory_fbff_vel,
                                    yawflip, 
                                    trajectory_fbff_yaw
                                    )

class DroneTask(Enum):
    HOVER = 'hover'
    YAWFLIP = 'yawflip'
    TRAJFBFF = 'trajectory_fbff'
    TRAJFBFF_VEL = 'trajectory_fbff_vel'
    TRAJFBFF_YAW = 'trajectory_fbff_yaw'

    def env(self):
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