import numpy as np
import torch
from quadsim_vision.utils.math_utils import *


from quadsim_vision.configuration import MPPIConfig, EnvConfig, DroneConfig
from quadsim_vision.utils.refs.base_ref import BaseRef
from quadsim_vision.utils.rigid_body import State_struct
from quadsim_vision.utils.timer import Timer

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class MPPIController():
	def __init__(self, env_config : EnvConfig, drone_config : DroneConfig, cntrl_config : MPPIConfig):
		# super().__init__(**kwargs)
		self.mppi_config = cntrl_config
		self.env_config = env_config
		self.drone_config = drone_config

		self.mppi_controller = MPPI_thrust_omega(env_config, drone_config, cntrl_config)
		self.f_t = np.zeros(3)
		self.runL1 = True

		self.prev_t = 0


	def _response(self, **response_inputs):
		t = response_inputs.get('t')
		state : State_struct = response_inputs.get('state')
		ref_dict : dict = response_inputs.get('ref')

		ref_func_obj : BaseRef = ref_dict['ref_obj']

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
		return action

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

class MPPI_thrust_omega():  
	'''
	using torch to sample and rollout
	a = [thrust (N), omega_des (rad/s)]

	config -> param.py 
	'''
	def __init__(self, env_config : EnvConfig, drone_config : DroneConfig, mppi_config : MPPIConfig):
		# self.mppi_config.Name = 'MPPI-thrust_omega'
		# self.env = env

		self.mppi_config = mppi_config
		self.env_config = env_config
		self.drone_config = drone_config

		self.thrust_hover = self.drone_config.mass * self.env_config.g

		# MPPI parameters
		self.t_H = np.arange(0, self.mppi_config.H * self.env_config.dt, self.env_config.dt)

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

		self.a_max[0] = self.drone_config.a_max[0] * 4

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
		dt = self.env_config.dt

		# u = actions[:, :, THRUST] / self.drone_config.mass
		# angvel = actions[:, :, ANGVEL]
		# print(self.u.size())
		# print(actions.size())
		# exit()

		self.u = self.u + self.env_config.k * (actions[:, :, THRUST] / self.drone_config.mass - self.u)
		self.angvel = self.angvel + self.env_config.k * (actions[:, :, ANGVEL] - self.angvel)

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
		accel = u.unsqueeze(2) * z_from_q(states[:, :, QUAT]) - self.env_config.g * e3.view(1, 1, 3) + L1_adapt[None, None, :]
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
		self.time_step = int(np.ceil(time / self.env_config.dt))
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

		cost *= self.env_config.dt
		self.timer.toc('reward')

		# compute weight
		self.timer.tic()
		cost -= torch.min(cost)
		Weight = torch.softmax(-cost / self.mppi_config.lam, dim=0)
		self.a_mean = torch.sum(actions * Weight.view(self.mppi_config.N, 1, 1), dim=0)

		self.timer.toc('update')

		# output the final command motorForce
		a_final = self.a_mean[0, :] # (4,)

		a_final[THRUST] = a_final[THRUST] / self.drone_config.mass # crazyflie takes care of the mass
		return a_final
	
	# def omega_controller(self, state, a):
	# 	# input:
	# 	#   state: tensor, (N, 13)
	# 	#   a (thrust and omega_des): tensor, (N, 4)
	# 	# output:
	# 	#   motorForce: tensor, (N, 4)
	# 	T_d = a[:, 0]
	# 	omega_d = a[:, 1:]
	# 	omega = state[:, 10:13]
	# 	omega_e = omega_d - omega

	# 	torque = self.mppi_config.omega_gain * omega_e # tensor, (N, 3)
	# 	torque = torch.mm(self.env.J, torque.T).T
	# 	torque -= torch.cross(torch.mm(self.env.J, omega.T).T, omega)

	# 	wrench = torch.cat((T_d.view(self.mppi_config.N,1), torque), dim=1) # tensor, (N, 4)
	# 	motorForce = torch.mm(self.env.B0_inv, wrench.T).T
	# 	motorForce = torch.clip(motorForce, self.env.a_min, self.env.a_max)
	# 	return motorForce



	# def rollout(self, startstate, actions):
	# 	"""
	# 	------ VECTORIZED DYNAMICS MODEL ------
	# 	converts actions to states
	# 	input:
	# 		startstate: (13,)
	# 		actions: (N, H, 4)
	# 	returns:
	# 		states: (N, H, 13)
	# 	"""
	# 	N, H, _ = actions.shape
	# 	xdim = startstate.shape[0]

	# 	e3 = torch.tensor((0, 0, 1))
	# 	dt = self.env_config.dt

	# 	self.u = self.u + self.env_config.k * (actions[:, :, THRUST] / self.drone_config.mass - self.u)
	# 	self.angvel = self.angvel + self.env_config.k * (actions[:, :, ANGVEL] - self.angvel)

	# 	u = self.u
	# 	angvel = self.angvel

	# 	states = torch.zeros(N, H, xdim)

	# 	self.timer.tic()
	# 	dang = angvel * dt
	# 	angnorm = torch.linalg.norm(dang, axis=2)
	# 	hangnorm = 0.5 * angnorm
	# 	axisfactor = torch.where(angnorm < 1e-8, 0.5 + angnorm ** 2 / 48, torch.sin(hangnorm) / angnorm).unsqueeze(2)

	# 	rotquat = torch.cat((torch.cos(hangnorm).unsqueeze(2), axisfactor * dang), dim=2)
	# 	quat = startstate[QUAT]
	# 	quat = quat.unsqueeze(0)

	# 	states = self.qmultiply_loop(quat, rotquat, states, H)

	# 	self.timer.toc('att kinematics')

	# 	self.timer.tic()
	# 	accel = u.unsqueeze(2) * z_from_q(states[:, :, QUAT]) - self.env_config.g * e3.view(1, 1, 3)
	# 	states[:, :, VEL] = startstate[VEL] + torch.cumsum(accel * dt, dim=1)
	# 	states[:, :, POS] = startstate[POS] + torch.cumsum(states[:, :, VEL] * dt, dim=1)
	# 	self.timer.toc('pos dynamics')

	# 	return states

	

	# def compute_cost(self, states):
	# 	self.timer.tic()

	# 	state_ref = self.env.ref_trajectory[:, self.time_step : self.time_step + self.mppi_config.H].T.unsqueeze(0)
	# 	# print(self.env.ref_func_t(0, 1))
	# 	# state_ref = torch.as_tensor(self.env.ref_func_t(self.t_H + self.time_step * self.env_config.dt), dtype=torch.float32).unsqueeze(0)

	# 	poserr = states[:, :, POS] - state_ref[:, :, POS]
	# 	# TODO, use discounting?
	# 	cost = self.mppi_config.alpha_p * torch.linalg.norm(poserr, dim=2) + \
	# 		   self.mppi_config.alpha_R * qdistance_torch(states[:, :, QUAT], state_ref[:, :, QUAT])
	# 	cost *= self.env_config.dt

	# 	self.timer.toc('reward')
	# 	return cost

	# def update(self, cost, actions):
	# 	# compute weight
	# 	self.timer.tic()
	# 	cost = cost.sum(dim=1)
	# 	cost -= torch.min(cost)
	# 	Weight = torch.softmax(-cost / self.mppi_config.lam, dim=0)
	# 	# Weight_mean = Weight.view(self.mppi_config.N, 1).repeat(1, 4) # tensor, (N,4)
	# 	# Weight_Sigma = Weight.view(self.mppi_config.N, 1, 1).repeat(1, 4, 4) # tensor, (N,4,4)

	# 	# self.a_mean = torch.einsum('bhi, b -> hi', A, Weight)
	# 	self.a_mean = torch.sum(actions * Weight.view(self.mppi_config.N, 1, 1), dim=0)
	# 	# update mean and Sigma, but just mean for now.
	# 	# for h in range(self.mppi_config.H):
	# 	# self.a_mean[h,:] = (1 - self.mppi_config.gamma_mean) * self.a_mean[h,:]
	# 	# self.a_Sigma[h,:,:] = (1 - self.mppi_config.gamma_Sigma) * self.a_Sigma[h,:,:]

	# 	# self.a_mean[h, :] = torch.sum(Weight_mean * A[:, h, :], dim=0)
	# 	# self.a_mean[h,:] += self.mppi_config.gamma_mean * new_mean

	# 	# m = A[:, h, :] - self.a_mean[h,:].view(1,4).repeat(self.mppi_config.N,1) # (N,4)
	# 	# new_Sigma = Weight_Sigma * torch.einsum("bi,bj->bij", m, m) # (N,4,4) einsum is for batch outer product
	# 	# new_Sigma = torch.sum(new_Sigma, dim=0)
	# 	# self.a_Sigma[h,:,:] += self.mppi_config.gamma_Sigma * new_Sigma
	# 	self.timer.toc('update')

	# def policy(self, state, time):
	# 	# input:
	# 	#   state: tensor, (13,)
	# 	# output:
	# 	#   motorForce: tensor, (4,)

	# 	# shift operator
	# 	self.shift()

	# 	# sample
	# 	self.timer.tic()
	# 	actions = self.sample()
	# 	self.timer.toc('sample')

	# 	# Use Dynamics Model to turn actions into states
	# 	#self.timer.tic()
	# 	self.time_step = int(np.ceil(time / self.env_config.dt))
	# 	states = self.rollout(state, actions)
	# 	#self.timer.toc('rollout')

	# 	# Evaluate States (Compute cost)
	# 	cost = self.compute_cost(states)

	# 	# Run the update
	# 	self.update(cost, actions)

	# 	# output the final command motorForce
	# 	a_final = self.a_mean[0, :] # (4,)
	# 	T_d = a_final[0]
	# 	omega_d = a_final[1:]
	# 	omega = state[10:13]
	# 	omega_e = omega_d - omega
	# 	torque = self.mppi_config.omega_gain * omega_e # tensor, (3,)
	# 	torque = torch.mv(self.env.J, torque)
	# 	torque -= torch.cross(torch.mv(self.env.J, omega), omega)
	# 	wrench = torch.cat((T_d.view(1), torque)) # tensor, (4,)
	# 	motorForce = torch.mv(self.env.B0_inv, wrench)
	# 	motorForce = torch.clip(motorForce, self.env.a_min, self.env.a_max)

	# 	return motorForce

	