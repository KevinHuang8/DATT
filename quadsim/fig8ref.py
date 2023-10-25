import numpy as np
from quadsim.flatref import StaticRef

class Fig8Ref:
    def __init__(self, a, altitude, period=2*np.pi):
        self.a = a
        self.altitude = altitude
        self.c = period / (2*np.pi)

    def pos(self, t):
        return np.array([
            self.a * np.sin(t / self.c),
            self.a * np.sin(t / self.c) * np.cos(t / self.c),
            t*0 + self.altitude
            ])

    def vel(self, t):
        return np.array([
            self.a / self.c * np.cos(t / self.c),
            self.a / self.c * (np.cos(t / self.c)**2 - np.sin(t / self.c)**2),
            t*0
        ])

    def acc(self, t):
        return np.array([
            -self.a / (self.c**2) * np.sin(t / self.c),
            -self.a / (self.c**2) * 4 * np.sin(t / self.c) * np.cos(t / self.c),
            t*0
        ])

    def jerk(self, t):
        return np.array([
            -self.a / (self.c**3) * np.cos(t / self.c),
            4 * self.a / (self.c**3) * (np.sin(t / self.c)**2 - np.cos(t / self.c)**2),
            t*0
        ])

    def snap(self, t):
        return np.array([
            self.a / (self.c**4) * np.sin(t / self.c),
            16 * self.a / (self.c**4) * np.sin(t / self.c) * np.cos(t / self.c),
            t*0
        ])

    def yaw(self, t):
        return 0
        #return t / self.c