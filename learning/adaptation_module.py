import os
import torch
import numpy as np
from DATT.learning.configs import *
from DATT.learning.utils.adaptation_network import AdaptationNetwork

class Adapation():
    def __init__(self, adaptation_type='l1', g=9.81, **kwargs):
        self.g = g
        self.A = -0.01
        self.dt = 0.02
        self.lamb = 0.1
        self.v_hat = np.zeros(3)
        self.d_hat = np.zeros(3)
        self.d_hat_t = np.zeros(3)

        # L1 adaptation
        if adaptation_type == 'l1':
            self.adaptation_step = self.l1_adaptation
        # Naive adaptation
        elif adaptation_type == 'naive':
            self.adaptation_step = self.naive_adaptation
        # Rapid Motor Adaptation RMA
        elif adaptation_type == 'rma':

            policy_name = kwargs.get('policy_name')
            e_dims = kwargs.get('e_dims')
            action_dims = 4
            trainenv = kwargs.get('trainenv') # DroneTask

            adapt_name = kwargs.get('adapt_name', None)

            if adapt_name is None:
                adapt_name = f'{policy_name}_adapt'

            if os.path.exists(SAVED_POLICY_DIR / f'{policy_name}_adapt' / f'{adapt_name}'):
                self.adaptation_network_state_dict = torch.load(SAVED_POLICY_DIR / f'{policy_name}_adapt' / f'{adapt_name}', map_location=torch.device('cpu'))
            elif os.path.exists(SAVED_POLICY_DIR / f'{adapt_name}'):
                self.adaptation_network_state_dict = torch.load(SAVED_POLICY_DIR / f'{adapt_name}', map_location=torch.device('cpu'))
            else:
                raise ValueError(f'Invalid adaptation network name: {adapt_name}')
            self.adaptation_network = AdaptationNetwork(input_dims=trainenv.base_dims + action_dims, e_dims=e_dims)
            
            
            self.adaptation_step = self.rma_adaptation

    def reset(self, ):
        self.v_hat = np.zeros(3)
        self.d_hat = np.zeros(3)
        self.d_hat_t = np.zeros(3)
        
    def l1_adaptation(self, v, f):
        unit_mass = 1
        g_vec = np.array([0, 0, -1]) * self.g
        alpha = 0.99
        phi = 1 / self.A * (np.exp(self.A * self.dt) - 1)

        a_t_hat = g_vec + f / unit_mass - self.d_hat_t + self.A * (self.v_hat - v)
        
        self.v_hat += a_t_hat * self.dt
        v_tilde = self.v_hat - v
        
        self.d_hat_t = 1 / phi * np.exp(self.A * self.dt) * v_tilde
        self.d_hat = -(1 - alpha) * self.d_hat_t + alpha * self.d_hat

        e_pred = self.d_hat

        return e_pred
    
    def naive_adaptation(self, v_t, f_t):
        unity_mass = 1
        a_t = (v_t - self.v_hat) / self.dt
        g_vec = np.array([0, 0, -1]) * self.g

        adapt_term = unity_mass * a_t - unity_mass * g_vec - f_t

        self.d_hat = (1 - self.lamb) * self.d_hat + self.lamb * adapt_term
        self.v_hat = v_t

        e_pred = self.d_hat

        return e_pred

        # print(self.d_hat)
    
    def pseudo_adaptation(self, v_t, f_t):
        pass

    def rma_adaptation(self, history):
        e_pred = self.adaptation_network(history)

        return e_pred