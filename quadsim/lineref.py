import numpy as np
from quadsim.flatref import StaticRef

class LineRef:
    def __init__(self, D, altitude, period=4):
        self.D = D
        self.altitude = altitude
        self.T = period

    def pos(self, t):
        if isinstance(t, np.ndarray):
            x = np.zeros_like(t)
            forward = (t // self.T) % 2 == 0
            back = (t // self.T) % 2 == 1
            x[forward] = self.D / self.T * (t[forward] % self.T)
            x[back] = self.D - (self.D / self.T * (t[back] % self.T))
        else:
            if (t // self.T) % 2 == 0:
                x = self.D / self.T * (t % self.T)
            else:
                x = self.D - (self.D / self.T * (t % self.T))
        return np.array([
            x,
            t*0,
            t*0 + self.altitude
            ])

    def vel(self, t):
        if isinstance(t, np.ndarray):
            x = np.zeros_like(t)
            x[(t // self.T) % 2 == 0] = self.D / self.T
            x[(t // self.T) % 2 == 1] = -self.D / self.T
        else:
            if (t // self.T) % 2 == 0:
                x = self.D / self.T
            else:
                x  = -self.D / self.T
        return np.array([
            x,
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