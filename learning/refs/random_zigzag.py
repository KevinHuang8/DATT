import numpy as np
import random
import matplotlib.pyplot as plt

class RandomZigzag:
    def __init__(self, max_D=np.array([1, 0, 0]), min_dt=0.6, max_dt=1.5, diff_axis=False, seed=2023, env_diff_seed=False, fixed_seed=False):
        self.max_D = max_D
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.diff_axis = diff_axis
        self.reset_count = 0
        self.env_diff_seed = env_diff_seed

        # print('seed', seed)
        self.seed = seed
        self.fixed_seed = fixed_seed
        np.random.seed(self.seed)
        self.reset()

    def reset(self):
        if self.fixed_seed: #self.reset_seed:
            np.random.seed(self.seed)
        elif self.env_diff_seed and self.reset_count > 0:
            seed = random.randint(0, 1000000)
            np.random.seed(seed)

        size = 100
        # keep same dt for all 3 dimensions for now
        if self.diff_axis:
            self.dt = np.random.uniform(self.min_dt, self.max_dt, size=(size, 3))
        else:
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
        self.reset_count += 1

    def calc_axis_i(self, t, axis_idx, vel=False):
        i = np.array(np.searchsorted(self.T[:, axis_idx], t))
        zero = i == 0
        left = np.array(self.x[i - 1, axis_idx])
        left[zero] = 0.0

        right = self.x[i, axis_idx]

        t_left = np.array(self.T[i - 1, axis_idx])
        t_left[zero] = 0.0
        t_right = self.T[i, axis_idx]

        left = left.T
        right = right.T
        x = left + (right - left) / (t_right - t_left) * (t - t_left)

        if vel:
            v = (right - left) / (t_right - t_left)
            return v
        
        return x

    def pos(self, t):
        if self.diff_axis:
            x = self.calc_axis_i(t, 0)
            y = self.calc_axis_i(t, 1)
            z = self.calc_axis_i(t, 2)
            return np.array([x, y, z])

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
        if self.diff_axis:
            vx = self.calc_axis_i(t, 0, vel=True)
            vy = self.calc_axis_i(t, 1, vel=True)
            vz = self.calc_axis_i(t, 2, vel=True)
            return np.array([vx, vy, vz])

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
        return t * 0

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
    ref = RandomZigzag(np.array([1, 1, 0.2]), 0.5, 1.5, diff_axis=True)
    t = np.linspace(0, 10, 100)
    x = ref.pos(t)
    print(x.shape)
    v = ref.vel(t)
    yaw = ref.yaw(t)
    plt.subplot(3, 1, 1)
    plt.plot(t, x[0])
    plt.plot(t, v[0])
    plt.subplot(3, 1, 2)
    plt.plot(t, x[1])
    plt.plot(t, v[1])
    plt.subplot(3, 1, 3)
    plt.plot(t, x[2])
    plt.plot(t, v[2])

    #plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[0], x[1], x[2])

    plt.show()
