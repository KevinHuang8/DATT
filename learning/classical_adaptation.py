import numpy as np

class Adapation():
    def __init__(self, adaptation_type='l1', g=9.81):
        self.g = g
        self.A = -0.01
        self.dt = 0.02
        self.lamb = 0.1
        self.v_hat = np.zeros(3)
        self.d_hat = np.zeros(3)
        self.d_hat_t = np.zeros(3)

        self.adaptation_step = self.l1_adaptation

    def reset(self, ):
        self.v_hat = np.zeros(3)
        self.d_hat = np.zeros(3)
        self.d_hat_t = np.zeros(3)
        
    def l1_adaptation(self, v, f):
        unit_mass = 1
        g_vec = np.array([0, 0, -1]) * self.g
        alpha = 0.98
        phi = 1 / self.A * (np.exp(self.A * self.dt) - 1)

        a_t_hat = g_vec + f / unit_mass - self.d_hat_t + self.A * (self.v_hat - v)
        
        self.v_hat += a_t_hat * self.dt
        v_tilde = self.v_hat - v
        
        self.d_hat_t = 1 / phi * np.exp(self.A * self.dt) * v_tilde
        self.d_hat = -(1 - alpha) * self.d_hat_t + alpha * self.d_hat

    def naive_adaptation(self, v_t, f_t):
        unity_mass = 1
        a_t = (v_t - self.v_hat) / self.dt
        g_vec = np.array([0, 0, -1]) * self.g

        adapt_term = unity_mass * a_t - unity_mass * g_vec - f_t

        self.d_hat = (1 - self.lamb) * self.d_hat + self.lamb * adapt_term
        self.v_hat = v_t

        print(self.d_hat)
    
    def pseudo_adaptation(self, v_t, f_t):
        pass