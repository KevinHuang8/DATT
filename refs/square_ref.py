import numpy as np
from DATT.refs.base_ref import BaseRef

class SquareRef(BaseRef):
    def __init__(self, altitude, D1, D2, T1, T2, **kwargs):
        offset_pos = kwargs.get('offset_pos', np.zeros(3))
        super().__init__(offset_pos)
        self.altitude = altitude
        self.D1 = D1
        self.D2 = D2
        self.T1 = T1
        self.T2 = T2

    def pos(self, t):
        if isinstance(t, np.ndarray):
            x = np.zeros_like(t)
            y = np.zeros_like(t)
            c1 = (t // self.T1) % 2 == 0
            c2 = (t // self.T1) % 2 == 1
            x[c1] = self.D1 / self.T1 * (t[c1] % self.T1)
            x[c2] = self.D1 - (self.D1 / self.T1 * (t[c2] % self.T1))


            c1 = (t // self.T2) % 4 == 0
            c2 = (t // self.T2) % 4 == 1
            c3 = (t // self.T2) % 4 == 2
            c4 = (t // self.T2) % 4 == 3

            y[c1] = self.D2 / self.T2 * (t[c1] % self.T2)
            y[c2] = self.D2 - (self.D2 / self.T2 * (t[c2] % self.T2))
            y[c3] = -self.D2 / self.T2 * (t[c3] % self.T2)
            y[c4] = -self.D2 + (self.D2 / self.T2 * (t[c4] % self.T2))
        else:
            if (t // self.T1) % 2 == 0:
                x = self.D1 / self.T1 * (t % self.T1)
            else:
                x = self.D1 - (self.D1 / self.T1 * (t % self.T1))


            if (t // self.T2) % 4 == 0:
                y = self.D2 / self.T2 * (t % self.T2)
            elif (t // self.T2) % 4 == 1:
                y = self.D2 - (self.D2 / self.T2 * (t % self.T2))
            elif (t // self.T2) % 4 == 2:
                y = -self.D2 / self.T2 * (t % self.T2)
            else:
                y = -self.D2 + (self.D2 / self.T2 * (t % self.T2))
        return np.array([
            x,
            y,
            t*0 + self.altitude
            ])

    def vel(self, t):
        if isinstance(t, np.ndarray):
            x = np.zeros_like(t)
            y = np.zeros_like(t)
            x[(t // self.T1) % 2 == 0] = self.D1 / self.T1
            x[(t // self.T1) % 2 == 1] = -self.D1 / self.T1
            
            y[(t // self.T2) % 4 == 0] = self.D2 / self.T2
            y[(t // self.T2) % 4 == 1] = -self.D2 / self.T2
            y[(t // self.T2) % 4 == 2] = -self.D2 / self.T2
            y[(t // self.T2) % 4 == 3] = self.D2 / self.T2
        else:
            if (t // self.T1) % 2 == 0:
                x = self.D1 / self.T1
            else:
                x  = -self.D1 / self.T1

            if (t // self.T2) % 4 == 0:
                y = self.D2 / self.T2
            elif (t // self.T2) % 4 == 1:
                y = -self.D2 / self.T2
            elif (t // self.T2) % 4 == 2:
                y = -self.D2 / self.T2
            else:
                y = self.D2 / self.T2
        return np.array([
            x,
            y,
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