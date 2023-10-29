import numpy as np
import random
from DATT.python_utils import polyu
from DATT.learning.refs.base_ref import BaseRef

class ChainedPolyRef(BaseRef):
    def __init__(self, altitude, use_y=False, min_dt=1.5, max_dt=4.0, degree=3, seed=2023, env_diff_seed=False, fixed_seed=False):
        assert degree % 2 == 1

        self.altitude = altitude
        self.degree = degree
        self.use_y = use_y
        self.seed = seed
        self.fixed_seed = fixed_seed
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.env_diff_seed = env_diff_seed

        self.reset_count = 0

        np.random.seed(seed)
        self.reset()

    def generate_coeffs(self, size, dt):
        b_values = np.random.uniform(-1.5, 1.5, size=((self.degree + 1) // 2, size + 1))
        b_values[0, :] = 0
        b_values = np.concatenate((b_values[:, :-1], b_values[:, 1:]), axis=0)

        A_values = np.zeros((self.degree + 1, self.degree + 1, size))
        coeffs = np.zeros((self.degree + 1, size))

        for i in range(size):
            A_values[:, :, i] = polyu.deriv_fitting_matrix(self.degree + 1, dt[i])
            coeffs[:, i] = np.linalg.solve(A_values[:, :, i], b_values[:, i])[::-1]

        return coeffs
    
    def reset(self):
        if self.fixed_seed:
            np.random.seed(self.seed)
        elif self.env_diff_seed and self.reset_count > 0:
            np.random.seed(random.randint(0, 1000000))

        size = 100        

        self.dt_x = np.random.uniform(self.min_dt, self.max_dt, size=(size))
        # simplifies evaluation, will error if evaluating at t > T max
        self.T_x = np.r_[np.cumsum(self.dt_x, axis=0), 0.0]
        self.x_coeffs = self.generate_coeffs(size, self.dt_x)
        if self.use_y:
            self.dt_y = np.random.uniform(self.min_dt, self.max_dt, size=(size))
            self.T_y = np.r_[np.cumsum(self.dt_y, axis=0), 0.0]
            self.y_coeffs = self.generate_coeffs(size, self.dt_y)

        self.reset_count += 1

    def pos(self, t):
        i_x = np.array(np.searchsorted(self.T_x, t))
        offset = self.T_x[i_x - 1]
        x = np.polyval(self.x_coeffs[:, i_x], t - offset)

        if self.use_y:
            i_y = np.searchsorted(self.T_y, t)
            offset = self.T_y[i_y - 1]
            y = np.polyval(self.y_coeffs[:, i_y], t - offset)
        else:
            y = t*0
        return np.array([
            x,
            y,
            t*0 + self.altitude
        ])

    def vel(self, t):
        i_x = np.searchsorted(self.T_x, t)
        offset = self.T_x[i_x - 1] 
        x = np.polyval(np.polyder(self.x_coeffs[:, i_x]), t - offset)
        if self.use_y:
            i_y = np.searchsorted(self.T_y, t)
            offset = self.T_y[i_y - 1] 
            y = np.polyval(np.polyder(self.y_coeffs[:, i_y]), t - offset)
        else:
            y = t*0 
        return np.array([
            x,
            y,
            t*0
            ])

    def acc(self, t):
        i_x = np.searchsorted(self.T_x, t)
        offset = self.T_x[i_x - 1] 
        x = np.polyval(np.polyder(self.x_coeffs[:, i_x], 2), t - offset)
        if self.use_y:
            i_y = np.searchsorted(self.T_y, t)
            offset = self.T_y[i_y - 1] 
            y = np.polyval(np.polyder(self.y_coeffs[:, i_y], 2), t - offset)
        else:
            y = t*0
        return np.array([
            x,
            y,
            t*0
        ])

    def jerk(self, t):
        i_x = np.searchsorted(self.T_x, t)
        offset = self.T_x[i_x - 1] 
        x = np.polyval(np.polyder(self.x_coeffs[:, i_x], 3), t - offset)
        if self.use_y:
            i_y = np.searchsorted(self.T_y, t)
            offset = self.T_y[i_y - 1] 
            y = np.polyval(np.polyder(self.y_coeffs[:, i_y], 3), t - offset)
        else:
            y = t*0
        return np.array([
            x,
            y,
            t*0
        ])

    def snap(self, t):
        i_x = np.searchsorted(self.T_x, t)
        offset = self.T_x[i_x - 1] 
        x = np.polyval(np.polyder(self.x_coeffs[:, i_x], 4), t - offset)
        if self.use_y:
            i_y = np.searchsorted(self.T_y, t)
            offset = self.T_y[i_y - 1] 
            y = np.polyval(np.polyder(self.y_coeffs[:, i_y], 4), t - offset)
        else:
            y = t*0
        return np.array([
            x,
            y,
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
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ref = ChainedPolyRef(altitude=1.0, use_y=True, seed=np.random.randint(0, 1000000), degree=3)
    t = np.linspace(0, 10, 500)

    plt.subplot(2, 1, 1)
    plt.plot(t, ref.pos(t)[0, :], label='x')
    plt.subplot(2, 1, 2)
    plt.plot(t, ref.pos(t)[1, :], label='y')
    plt.show()