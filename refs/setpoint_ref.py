import numpy as np
from DATT.refs.base_ref import BaseRef

class SetpointRef(BaseRef):
    def __init__(self, setpoint=(0, 0, 0), randomize=False, **kwargs):
        offset_pos = kwargs.get('offset_pos', np.zeros(3))
        super().__init__(offset_pos)
        self.randomize = randomize
        self.default = setpoint
        self.reset()

    def reset(self):
        if self.randomize:
            self.setpoint = np.array([
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1)
            ])
        else:
            self.setpoint = np.array(self.default)

    def pos(self, t):
        if isinstance(t, np.ndarray):
            return np.array([
            t*0 + self.setpoint[0],
            t*0 + self.setpoint[1],
            t*0 + self.setpoint[2]
            ])
        return self.setpoint
    
    def vel(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
            ])

    def acc(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def jerk(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def snap(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def yaw(self, t):
        if isinstance(t, np.ndarray):
            y = np.zeros_like(t)
            # y[(t // self.T) % 2 == 0] = 0
            # y[(t // self.T) % 2 == 1] = np.pi
        else:
            return 0
            # if (t // self.T) % 2 == 0:
            #     return 0
            # else:
            #     return np.pi
        return y

    def yawvel(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def yawacc(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])