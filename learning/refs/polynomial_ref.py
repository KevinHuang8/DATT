import numpy as np
import random
from DATT.python_utils import plotu, polyu

class PolyRef:
    def __init__(self, altitude, use_y=False, t_end=10.0, degree=3, seed=2023, env_diff_seed=False, fixed_seed=False):
        assert degree % 2 == 1

        self.altitude = altitude
        self.degree = degree
        self.t_end = t_end
        self.use_y = use_y
        self.seed = seed
        self.fixed_seed = fixed_seed
        self.env_diff_seed = env_diff_seed
        self.reset_count = 0

        np.random.seed(seed)
        self.reset()

    def generate_coeff(self):
        b = np.random.uniform(-1, 1, size=(self.degree + 1, ))
        b[0] = 0
        b[(self.degree + 1) // 2] = 0 

        A = polyu.deriv_fitting_matrix(self.degree + 1, self.t_end)

        return np.linalg.solve(A, b)[::-1]
    
    def reset(self):
        if self.fixed_seed:
            np.random.seed(self.seed)
        elif self.env_diff_seed and self.reset_count > 0:
            np.random.seed(random.randint(0, 1000000))

        self.x_coeff = self.generate_coeff()
        if self.use_y:
            self.y_coeff = self.generate_coeff()

        self.reset_count += 1

    def pos(self, t):
        x = np.polyval(self.x_coeff, t)
        if self.use_y:
            y = np.polyval(self.y_coeff, t)
        else:
            y = t*0
        return np.array([
            x,
            y,
            t*0 + self.altitude
    ])

    def vel(self, t):
        x = np.polyval(np.polyder(self.x_coeff), t)
        if self.use_y:
            y = np.polyval(np.polyder(self.y_coeff), t)
        else:
            y = t*0 
        return np.array([
            x,
            y,
            t*0
            ])

    def acc(self, t):
        x = np.polyval(np.polyder(self.x_coeff, 2), t)
        if self.use_y:
            y = np.polyval(np.polyder(self.y_coeff, 2), t)
        else:
            y = t*0
        return np.array([
            x,
            y,
            t*0
        ])

    def jerk(self, t):
        x = np.polyval(np.polyder(self.x_coeff, 3), t)
        if self.use_y:
            y = np.polyval(np.polyder(self.y_coeff, 3), t)
        else:
            y = t*0
        return np.array([
            x,
            y,
            t*0
        ])

    def snap(self, t):
        x = np.polyval(np.polyder(self.x_coeff, 4), t)
        if self.use_y:
            y = np.polyval(np.polyder(self.y_coeff, 4), t)
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

    ref = PolyRef(altitude=1.0, use_y=True, t_end=10.0, degree=3)
    t = np.linspace(0, 10, 500)

    plt.subplot(2, 1, 1)
    plt.plot(t, ref.pos(t)[0, :], label='x')
    plt.subplot(2, 1, 2)
    plt.plot(t, ref.pos(t)[1, :], label='y')
    plt.show()