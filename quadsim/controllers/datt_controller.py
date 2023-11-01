import numpy as np
import sys
import DATT
sys.modules['quadsim'] = DATT

# from DroneLearning.Controllers.ctrl_backbone import ControllerBackbone
import time
from DATT.quadsim.control import Controller
# from DATT.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR
# from DATT.learning.configs_enum import *
from DATT.learning.configs import *
from DATT.learning.tasks import DroneTask
from DATT.learning.refs import TrajectoryRef

from DATT.quadsim.models import RBModel
from DATT.quadsim.controllers.cntrl_config import DATTConfig
from DATT.learning.utils.adaptation_network import AdaptationNetwork
from DATT.configuration.configuration import AllConfig
from DATT.learning.adaptation_module import Adapation

from DATT.configuration.configuration import AllConfig


import torch

class DATTController(Controller):
	def __init__(self, config : AllConfig, cntrl_config : DATTConfig):
		super().__init__()

		self.datt_config = cntrl_config
		self.config = config

		self.algo = RLAlgo.PPO

		self.algo_class = self.algo.algo_class()
		self.env = self.datt_config.task.env()(
			config=self.config,
		)
		
		self.count = 0
		self.v_prev = 0
		self.prev_t = 0

		self.policy = self.algo_class.load(SAVED_POLICY_DIR / f'{self.datt_config.policy_name}', self.env)
		if self.datt_config.adaptive == True:
			
			edim = self.config.adapt_config.get_e_dim()
			self.adaptation_module = Adapation(self.datt_config.adaptation_type, 
											self.config.sim_config.g, 
											policy_name = self.datt_config.adaptive_policy_name,
											adapt_name = self.datt_config.adaptive_policy_name,
											e_dims = edim,
											trainenv = self.env)

		self.prev_pos = 0.

		self.history = np.zeros((1, 14, 5))
		self.history_rma = np.zeros((1, 14, 50))
    
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

		if self.config.training_config.body_frame:
			pos = rot.inv().apply(pos)
			vel = rot.inv().apply(vel)

		obs_ = np.hstack((pos, vel, quat))
    

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
			obs_ = np.hstack((obs_, wind_terms))

				
		if self.config.policy_config.ff_term is False:
			pass
		else:
			obs_ = np.hstack([obs_, obs_[0:3] - rot.inv().apply(self.ref_func.pos(t))] + [obs_[0:3] - rot.inv().apply(self.ref_func.pos(t + 3 * i * dt)) for i in range(self.config.policy_config.time_horizon)])

		action, _ = self.policy.predict(obs_, deterministic=True)

		rma_adaptation_input = np.concatenate((obs_bf, action), axis=0)
		self.history_rma = np.concatenate((rma_adaptation_input[None, :, None], self.history_rma[:, :, :-1]), axis=2)

		action[0] += self.config.sim_config.g()
		
		adaptation_input = np.concatenate((obs_bf, action), axis=0)
		self.history = np.concatenate((adaptation_input[None, :, None], self.history[:, :, :-1]), axis=2)

		self.count += 1
		self.v_prev = state.vel

		return action[0], action[1:]
