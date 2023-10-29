import numpy as np
import torch
from quadsim_vision.utils.math_utils import *


from DATT.quadsim.controllers.cntrl_config import MPPIConfig
from DATT.learning.refs.base_ref import BaseRef
from DATT.quadsim.rigid_body import State_struct
from quadsim_vision.utils.timer import Timer
from DATT.quadsim.control import Controller

from DATT.quadsim.models import RBModel


torch.set_default_tensor_type('torch.cuda.FloatTensor')

STD_MAX = 1e3
STD_MIN = 1e-6

# state inds
POS = slice(0, 3)
VEL = slice(3, 6)
QUAT = slice(6, 10)
OMEGA = slice(10, 13)

# action inds
THRUST = 0
ANGVEL = slice(1, 4)

class MPPIController(Controller):
	def __init__(self, model : RBModel, cntrl_config : MPPIConfig):
		super().__init__()
		# super().__init__(**kwargs)
		self.model = model
		self.mppi_config = cntrl_config
		# self.env_config = env_config
		# self.drone_config = drone_config
		self.mppi_controller = MPPI_thrust_omega(model, cntrl_config)
		self.f_t = np.zeros(3)
		self.runL1 = True

		self.prev_t = 0


	def response(self, **response_inputs):
		t = response_inputs.get('t')
		state : State_struct = response_inputs.get('state')
		# ref_dict : dict = response_inputs.get('ref')

		ref_func_obj = self.ref_func

		pos = state.pos
		vel = state.vel
		rot = state.rot
		quat = rot.as_quat()
		# (x,y,z,w) -> (w,x,y,z)
		quat = np.roll(quat, 1)

		obs = np.hstack((pos, vel, quat))
		state_torch = torch.as_tensor(obs, dtype=torch.float32)
		# L1_adapt = torch.zeros(3)
		# if self.runL1 and not self.pseudo_adapt and fl!=0:
		#   self.L1_adaptation(self.dt, state.vel, self.f_t)
		#   self.adaptation_terms[1:] = self.wind_adapt_term
		#   L1_adapt = torch.as_tensor(self.wind_adapt_term, dtype=torch.float32)
		# # action = self.mppi_controller.policy_cf(state=state_torch, time=t).cpu().numpy()
		# # start = time.time()

		# if self.pseudo_adapt:
		action = self.mppi_controller.policy_with_ref_func(state=state_torch, time=t, new_ref_func = ref_func_obj).cpu().numpy()
		# else:
		# action = self.mppi_controller.policy_with_ref_func(state=state_torch, time=t, new_ref_func=self.ref_func_t, L1_adapt=L1_adapt).cpu().numpy()
		# print(time.time() - start)
		# MPPI controller designed for output in world frame
		# World Frame -> Body Frame
		
		# st = time.time()
		action[1:] = (rot.as_matrix().T).dot(action[1:])

		self.f_t = rot.apply(np.array([0, 0, action[0]]))

		self.prev_t = t
		return action[THRUST], action[ANGVEL]



class MPPI_thrust_omega():  
	'''
	using torch to sample and rollout
	a = [thrust (N), omega_des (rad/s)]

	config -> param.py 
	'''
	def __init__(self, model : RBModel, mppi_config : MPPIConfig):
		# self.mppi_config.Name = 'MPPI-thrust_omega'
		# self.env = env

		self.mppi_config = mppi_config
		self.model = model

		self.thrust_hover = self.model.mass * self.model.g

		# MPPI parameters
		self.t_H = np.arange(0, self.mppi_config.H * self.model.dt, self.model.dt)

		self.u      = torch.zeros((self.mppi_config.N, self.mppi_config.H))
		self.angvel = torch.zeros((self.mppi_config.N, self.mppi_config.H, 3))
		self.a_mean = torch.zeros(self.mppi_config.H, 4) # mean of actions: tensor, (H, 4)
		self.a_mean[:, 0] = self.thrust_hover


		sample_std = self.mppi_config.sample_std
		self.sampling_std = torch.tensor((
		sample_std[0] * self.thrust_hover,
		sample_std[1],
		sample_std[2],
		sample_std[3],
		))

		self.time_step = 0

		self.a_min = torch.tensor(self.mppi_config.a_min)
		self.a_max = torch.tensor(self.mppi_config.a_max)

		self.a_max[0] = self.model.a_max[0] * 4

		# timer for code profiling
		self.timer = Timer(topics=['shift', 'sample', 'rollout', 'update', 'att kinematics', 'reward', 'pos dynamics'])
		# self.iscpp = 0
		# if self.iscpp:
		# 	torch.ops.load_library("/home/rwik/rwik_hdd/drones/Quadsim/quadsim/MPPI/libmppi_rollout.so")
		# 	self.qmultiply_loop = torch.ops.my_ops.mppi_rollout_qmultiplyloop_cuda
		# else:
		self.qmultiply_loop = self.qmultiply_loop_python
			# self.cpp_mppi_qmultiply_cpu = torch.ops.my_ops.mppi_rollout_qmultiplyloop


	def sample(self):
		return torch.normal(mean=self.a_mean.repeat(self.mppi_config.N, 1, 1), std=self.sampling_std)

	
	def qmultiply_loop_python(self, quat, rotquat, states, H):
		for h in range(self.mppi_config.H):
			# Switch order to make angvel in body frame.
			states[:, h, QUAT] = qmultiply_torch(rotquat[:, h], quat)
			quat = states[:, h, QUAT]
		
		return states
	
	def rollout_with_adaptation(self, startstate, actions, L1_adapt=torch.zeros(3)):
		"""
		------ VECTORIZED DYNAMICS MODEL ------
		converts actions to states
		input:
			startstate: (13,)
			actions: (N, H, 4)
		returns:
			states: (N, H, 13)
		"""
		N, H, _ = actions.shape
		xdim = startstate.shape[0]

		e3 = torch.tensor((0, 0, 1))
		dt = self.model.dt

		# u = actions[:, :, THRUST] / self.drone_config.mass
		# angvel = actions[:, :, ANGVEL]
		# print(self.u.size())
		# print(actions.size())
		# exit()

		self.u = self.u + self.model.k * (actions[:, :, THRUST] / self.model.mass - self.u)
		self.angvel = self.angvel + self.model.k * (actions[:, :, ANGVEL] - self.angvel)

		u = self.u
		angvel = self.angvel

		states = torch.zeros(N, H, xdim)

		self.timer.tic()
		dang = angvel * dt
		angnorm = torch.linalg.norm(dang, axis=2)
		hangnorm = 0.5 * angnorm
		axisfactor = torch.where(angnorm < 1e-8, 0.5 + angnorm ** 2 / 48, torch.sin(hangnorm) / angnorm).unsqueeze(2)

		rotquat = torch.cat((torch.cos(hangnorm).unsqueeze(2), axisfactor * dang), dim=2)
		quat = startstate[QUAT]
		quat = quat.unsqueeze(0)

		states = self.qmultiply_loop(quat, rotquat, states, H)

		self.timer.toc('att kinematics')
		
		self.timer.tic()
		accel = u.unsqueeze(2) * z_from_q(states[:, :, QUAT]) - self.model.g * e3.view(1, 1, 3) + L1_adapt[None, None, :]
		states[:, :, VEL] = startstate[VEL] + torch.cumsum(accel * dt, dim=1)
		states[:, :, POS] = startstate[POS] + torch.cumsum(states[:, :, VEL] * dt, dim=1)
		self.timer.toc('pos dynamics')

		return states

	def shift(self):
		self.timer.tic()
		a_mean_old = self.a_mean.clone()
		#a_Sigma_old = self.a_Sigma.clone()
		self.a_mean[:-1,:] = a_mean_old[1:,:]
		#self.a_Sigma[:-1,:,:] = a_Sigma_old[1:,:,:]
		self.timer.toc('shift')
	
	def policy_with_ref_func(self, state, time, new_ref_func=None, L1_adapt=torch.zeros(3)):
		'''
		Policy compatible with crazyflie stack.
		'''
		# input:
		#   state: tensor, (13,)
		# output:
		#   action: tensor, (4,)  [thurst, angular velocities(world frame)]

		# shift operator
		self.timer.tic()
		a_mean_old = self.a_mean.clone()
		#a_Sigma_old = self.a_Sigma.clone()
		self.a_mean[:-1,:] = a_mean_old[1:,:]
		#self.a_Sigma[:-1,:,:] = a_Sigma_old[1:,:,:]
		self.timer.toc('shift')

		# sample
		self.timer.tic()
		actions = self.sample()
		self.timer.toc('sample')

		# Use Dynamics Model to turn actions into states
		self.timer.tic()
		self.time_step = int(np.ceil(time / self.model.dt))
		# states = self.rollout(state, actions)
		states = self.rollout_with_adaptation(state, actions, L1_adapt)
		self.timer.toc('rollout')

		# Evaluate States (Compute cost)
		self.timer.tic()

		state_ref = torch.as_tensor(new_ref_func.ref_vec(self.t_H + time).T, dtype=torch.float32).unsqueeze(0)
		# state_ref = self.env.ref_trajectory[:, self.time_step : self.time_step + self.mppi_config.H].T.unsqueeze(0)
		# state_ref = torch.as_tensor(self.env.ref_func_t(self.t_H + time), dtype=torch.float32).unsqueeze(0)

		poserr = states[:, :, POS] - state_ref[:, :, POS]
		# TODO, use discounting?
		cost = self.mppi_config.alpha_p * torch.sum(torch.linalg.norm(poserr, dim=2), dim=1) + \
			self.mppi_config.alpha_R * torch.sum(qdistance_torch(states[:, :, QUAT], state_ref[:, :, QUAT]), dim=1)

		cost *= self.model.dt
		self.timer.toc('reward')

		# compute weight
		self.timer.tic()
		cost -= torch.min(cost)
		Weight = torch.softmax(-cost / self.mppi_config.lam, dim=0)
		self.a_mean = torch.sum(actions * Weight.view(self.mppi_config.N, 1, 1), dim=0)

		self.timer.toc('update')

		# output the final command motorForce
		a_final = self.a_mean[0, :] # (4,)

		a_final[THRUST] = a_final[THRUST] / self.model.mass # crazyflie takes care of the mass
		return a_final	