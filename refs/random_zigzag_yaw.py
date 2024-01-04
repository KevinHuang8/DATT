import numpy as np
import matplotlib.pyplot as plt
from DATT.refs.base_ref import BaseRef

class RandomZigzagYaw(BaseRef):
    def __init__(self, max_D=np.array([1, 0, 0]), min_dt=0.9, max_dt=1.5, seed=2023, fixed_seed=False, **kwargs):
        offset_pos = kwargs.get('offset_pos', np.zeros(3))
        super().__init__(offset_pos)
        self.max_D = max_D
        self.min_dt = min_dt
        self.max_dt = max_dt

        # print('seed', seed)
        self.seed = seed
        self.fixed_seed = fixed_seed
        np.random.seed(self.seed)
        self.reset()

    def reset(self):
        if self.fixed_seed: #self.reset_seed:
            np.random.seed(self.seed)

        size = 100
        # keep same dt for all 3 dimensions for now
        self.dt = np.random.uniform(self.min_dt, self.max_dt, size=(size))
        self.T = np.cumsum(self.dt, axis=0)
        pos_high_x = np.random.uniform(0, self.max_D[0], size=(size // 2, 1))
        pos_low_x = np.random.uniform(-self.max_D[0], 0, size=(size // 2, 1))
        pos_high_y = np.random.uniform(0, self.max_D[1], size=(size // 2, 1))
        pos_low_y = np.random.uniform(-self.max_D[1], 0, size=(size // 2, 1))
        pos_high_z = np.random.uniform(0, self.max_D[2], size=(size // 2, 1))
        pos_low_z = np.random.uniform(-self.max_D[2], 0, size=(size // 2, 1))

        pos_high = np.hstack((pos_high_x, pos_high_y, pos_high_z))
        pos_low = np.hstack((pos_low_x, pos_low_y, pos_low_z))

        self.x = np.empty((size, 3), dtype=pos_high.dtype)
        self.x[0::2] = pos_high
        self.x[1::2] = pos_low

        # self.x = -self.x 

    def pos(self, t):
        i = np.array(np.searchsorted(self.T, t))

        zero = i == 0
        left = np.array(self.x[i - 1])
        left[zero] = 0.0
        right = self.x[i]

        t_left = np.array(self.T[i - 1])
        t_left[zero] = 0.0
        t_right = self.T[i] 

        left = left.T
        right = right.T

        x = left + (right - left) / (t_right - t_left) * (t - t_left)

        return x
    
    def vel(self, t):
        i = np.array(np.searchsorted(self.T, t))

        zero = i == 0
        left = np.array(self.x[i - 1])
        left[zero] = 0.0
        right = self.x[i]

        t_left = np.array(self.T[i - 1])
        t_left[zero] = 0.0
        t_right = self.T[i] 

        left = left.T
        right = right.T

        v = (right - left) / (t_right - t_left)

        return v
    
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
        i = np.array(np.searchsorted(self.T, t))

        if isinstance(t, np.ndarray):
            yaw = np.zeros_like(t)
            yaw[i % 2 == 0] = 0
            yaw[i % 2 == 1] = np.pi
        else:
            if i % 2 == 0:
                yaw = 0
            else:
                yaw = np.pi
        return yaw

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

if __name__ == "__main__":
    ref = RandomZigzagYaw(np.array([1, 0.0, 0]), 0.5, 1.5)
    t = np.linspace(0, 10, 100)
    x = ref.pos(t)
    v = ref.vel(t)
    yaw = ref.yaw(t)
    plt.subplot(4, 1, 1)
    plt.plot(t, x[0])
    plt.plot(t, v[0])
    plt.subplot(4, 1, 2)
    plt.plot(t, x[1])
    plt.plot(t, v[1])
    plt.subplot(4, 1, 3)
    plt.plot(t, x[2])
    plt.plot(t, v[2])
    plt.subplot(4, 1, 4)
    plt.plot(t, yaw)
    plt.show()
