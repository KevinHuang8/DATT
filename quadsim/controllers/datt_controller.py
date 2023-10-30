import numpy as np
import sys
import DATT
sys.modules['quadsim'] = DATT

# from DroneLearning.Controllers.ctrl_backbone import ControllerBackbone
import time
from DATT.quadsim.control import Controller
# from DATT.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR
from DATT.learning.configs_enum import *

from DATT.quadsim.models import RBModel
from DATT.quadsim.controllers.cntrl_config import DATTConfig
from DATT.learning.utils.adaptation_network import AdaptationNetwork
from DATT.learning.configuration.configuration import AllConfig
from DATT.learning.adaptation_module import Adapation

import torch

class DATTController(Controller):
	def __init__(self, model : RBModel, cntrl_config : DATTConfig):
		super().__init__()

		self.model = model
		self.datt_config = cntrl_config

		# self.set_policy()
		# self.select_policy_configs()

		self.algo = RLAlgo.PPO
		# self.config = None

		self.algo_class = self.algo.algo_class()
		# config = import_config(self.datt_config.config_filename)
		self.env = self.datt_config.task.env()(
			config=self.datt_config.config,
		)
		
		self.count = 0
		self.v_prev = 0
		self.prev_t = 0

		self.policy = self.algo_class.load(SAVED_POLICY_DIR / f'{self.datt_config.policy_name}', self.env)
		if self.datt_config.adaptive == True:
			
			edim = self.datt_config.config.adapt_config.get_e_dim()
			self.adaptation_module = Adapation(self.datt_config.adaptation_type, 
											self.model.g, 
											policy_name = self.datt_config.adaptive_policy_name,
											adapt_name = self.datt_config.adaptive_policy_name,
											e_dims = edim,
											trainenv = self.env)
			
			# self.adaptive_policy = AdaptationNetwork(14, edim, complex=True)
			# self.adaptive_policy.load_state_dict(torch.load(SAVED_POLICY_DIR / f'{self.datt_config.adaptive_policy_name}', map_location='cuda:0'))
		
		
		self.prev_pos = 0.

		self.history = np.zeros((1, 14, 5))
		self.history_rma = np.zeros((1, 14, 50))
    
    # Override L1 params
    # # naive params
    # self.lamb = 0.2

    # # L1 params
    # self.runL1 = False # L1 v/s naive toggle
    # self.filter_coeff = 5
    # self.A = -0.2
    # self.count = 0
  # def select_policy_configs(self,):
  #     # policy_dict = select_policy_config_(self.policy_config)
  #     self.task = self.datt_config.task
  #     self.policy_name = self.datt_config.policy_name
  #     self.config_filename = policy_dict["config_filename"]
  #     self.adaptive_policy_name = policy_dict["adaptive_policy_name"]
  #     self.body_frame = policy_dict["body_frame"]
  #     self.relative = policy_dict["relative"]
  #     self.log_scale = policy_dict["log_scale"]
  #     self.e_dims = policy_dict["e_dims"]
  #     self.u_struct = policy_dict["u_struct"]
  #     self.adaptation_warmup = policy_dict['adaptation_warmup']

	def response(self, fl = 1, **response_inputs):
		t = response_inputs.get('t')
		state = response_inputs.get('state')
		# ref_func = response_inputs.get('ref_func')

		if self.prev_t is None:
			dt = 0.02
		else:
			dt = t - self.prev_t

		if fl:
			self.prev_t = t

		# States
		pos = state.pos
		vel = state.vel
		rot = state.rot

		# Acceleration Estimation
		v_t = state.vel
		if self.count > 2:
			v_t = state.vel
			a_t = (v_t - self.v_prev) / dt
		else:
			a_t = np.array([0, 0, 0]) 

		# Previous thrust action. f_t is in m / s**2
		unity_mass = 1
		f_t = rot.apply(np.array([0, 0, self.history[0, 10, 0]])) * unity_mass

		quat = rot.as_quat() 

		obs_bf = np.hstack((pos, vel, quat))

		if self.datt_config.config.training_config.body_frame:
			pos = rot.inv().apply(pos)
			vel = rot.inv().apply(vel)

		obs_ = np.hstack((pos, vel, quat))
    
    # st = time.time()

		if self.datt_config.adaptive : 
			if self.count > 2:
				v_t = state.vel
				a_t = (v_t - self.v_prev) / dt
			else:
				a_t = np.array([0, 0, 0])
			
			if self.datt_config.adaptation_type == 'rma':
				wind_terms = self.adaptation_module.adaptation_step(torch.tensor(self.history_rma).float())
				wind_terms = wind_terms[0].detach().cpu().numpy()
			elif self.datt_config.adaptation_type == 'l1':
				wind_terms = self.adaptation_module.adaptation_step(v_t, f_t)
			# wind_terms = np.zeros(3)
			obs_ = np.hstack((obs_, wind_terms))

				
    # if self.pseudo_adapt== False and fl!=0.0:
    #   if self.count > 2:
    #     v_t = state.vel
    #     a_t = (v_t - self.v_prev) / dt
    #   else:
    #     a_t = np.array([0, 0, 0])        
      
    #   if self.runL1:
    #     # L1 adaptation update
    #     self.L1_adaptation(dt, v_t, f_t)
    #   else:
    #     self.naive_adaptation(a_t, f_t)
      

    #   # rwik = self.adaptive_policy(torch.tensor(self.history_rma).float())

    #   # self.wind_adapt_term = rwik[0].detach().cpu().numpy()
    #   # print(self.wind_adapt_term)
    #   # print(rwik)
    #   # print(self.wind_adapt_term)
    #   # self.wind_adapt_term = np.random.normal(0, 0.5, self.e_dims) + np.array([1.0, -0.5, 0])
    #   self.adaptation_terms[1: ] = self.wind_adapt_term
    #   obs_ = np.hstack((obs, self.wind_adapt_term))
    # else:
    #   pseudo_adapt_term =  np.zeros(self.e_dims)
    #   # pseudo_adapt_term = np.random.normal(0, 0.5, self.e_dims) + np.array([1.0, -0.5, 0])
    #   self.adaptation_terms[1: ] = pseudo_adapt_term
    #   obs_ = np.hstack((obs, -pseudo_adapt_term))
    # mid = time.time() - st
    # import pdb;pdb.set_trace()
		if self.datt_config.config.policy_config.ff_term is False:
			pass
		else:
			obs_ = np.hstack([obs_, obs_[0:3] - rot.inv().apply(self.ref_func.pos(t))] + [obs_[0:3] - rot.inv().apply(self.ref_func.pos(t + 3 * i * dt)) for i in range(self.datt_config.config.policy_config.time_horizon)])

		#     else:
		#       ff_terms = [ref_func(t + 3 * i * dt)[0].pos for i in range(self.time_horizon)]
		#       obs_ = np.hstack([obs_, obs_[0:3] - ref_func(t)[0].pos] + ff_terms)

		action, _ = self.policy.predict(obs_, deterministic=True)

		rma_adaptation_input = np.concatenate((obs_bf, action), axis=0)
		self.history_rma = np.concatenate((rma_adaptation_input[None, :, None], self.history_rma[:, :, :-1]), axis=2)
		# # adaptation_input = torch.from_numpy(adaptation_input).to("cuda:0").float()

		#   # import pdb;pdb.set_trace()
		# if self.log_scale:
		#   action[0] = np.sinh(action[0])
		# else:
		action[0] += self.model.g
		
		adaptation_input = np.concatenate((obs_bf, action), axis=0)
		# if fl!=0.0:
		self.history = np.concatenate((adaptation_input[None, :, None], self.history[:, :, :-1]), axis=2)

		self.count += 1
		self.v_prev = state.vel

		# new_obs = np.hstack((obs, action))
		# self.obs_history[1:] = self.obs_history[0:-1]
		# self.obs_history[0] = new_obs
		# import pdb;pdb.set_trace()
		return action[0], action[1:]
