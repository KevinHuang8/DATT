import numpy as np

from quadsim.learning.configuration.configuration import *

drone_config = DroneConfiguration(
    mass = ConfigValue[float](1.0, randomize=False),
    I = ConfigValue[float](1.0, randomize=False),
    g = ConfigValue[float](9.8, False)
)

wind_config = WindConfiguration(
    is_wind = True,
    dir = ConfigValue[np.ndarray](
        default=np.zeros(3), 
        randomize=True,
        min=np.array([-2, -2, -2]),
        max=np.array([2, 2, 2])
    ),
    random_walk=ConfigValue[bool](True)
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
    k=ConfigValue[float](default=0.4, randomize=False),
    L1_simulation=ConfigValue[float](default=True, randomize=False),
)

adapt_config = AdaptationConfiguration(
    include = [EnvCondition.WIND]
)

train_config = TrainingConfiguration()

policy_config = PolicyConfiguration()

ref_config = RefConfiguration(
    init_ref = ConfigValue[int](1)
)

config = AllConfig(drone_config, wind_config, init_config, sim_config, adapt_config, train_config, policy_config, ref_config)


