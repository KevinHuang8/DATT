import numpy as np
from DATT.quadsim.flatref import StaticRef
from DATT.refs.base_ref import BaseRef

class CircleRef(BaseRef):
    def __init__(self, rad, altitude, offset=(1,0), period=2*np.pi, **kwargs):
        offset_pos = kwargs.get('offset_pos', np.zeros(3))
        super().__init__(offset_pos)
        self.radius = rad
        self.offset = offset
        self.altitude = altitude
        self.c = period / (2*np.pi)

    def pos(self, t):
        return np.array([
            -self.radius * np.cos(t / self.c) + self.offset[0],
            self.radius * np.sin(t / self.c) + self.offset[1],
            t*0 + self.altitude
            ])

    def vel(self, t):
        return np.array([
            self.radius / self.c * np.sin(t / self.c),
            self.radius / self.c * np.cos(t / self.c),
            t*0
            ])

    def acc(self, t):
        return np.array([
            self.radius / (self.c**2) * np.cos(t / self.c),
            -self.radius / (self.c**2) * np.sin(t / self.c),
            t*0
        ])

    def jerk(self, t):
        return np.array([
            -self.radius / (self.c**3) * np.sin(t / self.c),
            -self.radius / (self.c**3) * np.cos(t / self.c),
            t*0
        ])

    def snap(self, t):
        return np.array([
            -self.radius / (self.c**4) * np.cos(t / self.c),
            self.radius / (self.c**4) * np.sin(t / self.c),
            t*0
        ])

    def yaw(self, t):
        return 0
        #return t / self.c

if __name__ == '__main__':
    ref = CircleRef(altitude=0, rad=1.0, period=2.0)
    t = np.linspace(0, 10, 500)
    import matplotlib.pyplot as plt
    plt.subplot(2, 1, 1)
    plt.plot(t, ref.pos(t)[0, :], label='x')
    plt.subplot(2, 1, 2)
    plt.plot(t, ref.pos(t)[1, :], label='y')
    plt.show()