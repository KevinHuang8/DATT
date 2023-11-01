import numpy as np

from DATT.configuration.configuration import *

drone_config = DroneConfiguration(
    mass = ConfigValue[float](1.0, randomize=False, min=0.7, max=1.3),
    I = ConfigValue[float](1.0, randomize=False, min=0.9, max=1.1),
    # g = ConfigValue[float](9.8, False)
)

wind_config = WindConfiguration(
    is_wind = True,
    dir = ConfigValue[np.ndarray](
        default=np.zeros(3), 
        randomize=True,
        min=np.array([-1, -1, -1]),
        max=np.array([1, 1, 1])
    ),
)

init_config = InitializationConfiguration(
    pos = ConfigValue[np.ndarray](
        default=np.array([0.0, 0.0, 0.0]), 
        randomize=False,
        min=np.array([-0, -0, -0]),
        max=np.array([0, 0, 0])
    ),
    vel = ConfigValue[np.ndarray](
        default=np.array([0.0, 0.0, 0.0]), 
        randomize=False
    ),
    # Represented as Euler ZYX in degrees
    rot = ConfigValue[np.ndarray](
        default=np.array([0.0, 0.0, 0.0]),
        randomize=False
    ),
    ang = ConfigValue[np.ndarray](
        default=np.array([0.0, 0.0, 0.0]),
        randomize=False
    )
)

sim_config = SimConfiguration(
    linear_var=ConfigValue[float](default=0.0, randomize=False),
    angular_var=ConfigValue[float](default=0.0, randomize=False),
    obs_noise=ConfigValue[float](default=0.005, randomize=False),
    latency=ConfigValue[int](default=0.0, randomize=False),
    k=ConfigValue[float](default=0.4, randomize=False)
)

adapt_config = AdaptationConfiguration(
    include = [EnvCondition.WIND]
)

config = AllConfig(drone_config, wind_config, init_config, sim_config, adapt_config)


