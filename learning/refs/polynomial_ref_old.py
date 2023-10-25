import numpy as np
from DATT.quadsim.flatref import StaticRef

class PolynomialRef:
    def __init__(self, A, B, C, period, scale=1.25, altitude=0.75):
        self.A = A
        self.B = B
        self.C = C
        self.scale = scale
        self.altitude = altitude
        self.T = period

    def pos(self, t):
        tt = t / self.T
        x = self.scale * tt * (tt - self.A[0]) * (tt - self.B[0]) * (tt - self.C[0]) 
        y = self.scale * tt * (tt - self.A[1]) * (tt - self.B[1]) * (tt - self.C[1]) 
        z = self.scale / 4 * tt * (tt - self.A[2]) * (tt - self.B[2]) * (tt - self.C[2]) + self.altitude

        x = np.clip(x, -1.5, 1.5)
        y = np.clip(y, -1.5, 1.5)
        z = np.clip(z, 0.5, self.altitude + 0.5)

        return np.array([
            x,
            y,
            z
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