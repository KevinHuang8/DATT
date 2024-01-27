import time

import numpy as np

from scipy.spatial.transform import Rotation as R

from DATT.python_utils.timeseriesu import TimeSeries
from DATT.python_utils.mathu import e3

from DATT.quadsim.rigid_body import RigidBody
from DATT.quadsim.visualizer import Vis
from DATT.learning.utils.heap import heap

class QuadSim:
  def __init__(self, model, force_limit=200, torque_limit=100, angvel_limit=40, vis=False):
    self.gvec = np.array((0, 0, -model.g))
    self.mass = model.mass
    self.rb = RigidBody(mass=model.mass, inertia=model.I)
    self.model = model

    self.force_limit = force_limit # Newtons (N)
    # L_2 limit on vector
    self.torque_limit = torque_limit # Nm
    # Only used if step_angvel is called.
    self.angvel_limit = angvel_limit

    self.bodyz_force_des = 0.0
    self.angvel_des = np.zeros(3)
    self.curr_bodyz_force = 0.0
    self.curr_angvel = np.zeros(3)
    self.k = -0.9
    self.command_queue = heap()
    self.cmd_id = 0

    self.curr_torque = np.zeros(3)
    self.torque_des = np.zeros(3)

    self.vis = None
    if vis:
      self.vis_speed = 1.5
      self.vis = Vis()

  def getstate(self):
    return self.rb.state()

  def setstate(self, state):
    self.rb.setstate(state)

  def forcetorque_from_u(self, u, **kwargs):
    return u

  def reset(self):
    self.command_queue = heap()
    self.cmd_id = 0
    self.bodyz_force_des = 0.0
    self.angvel_des = np.zeros(3)
    self.curr_bodyz_force = 0.0
    self.curr_angvel = np.zeros(3)

    self.curr_torque = np.zeros(3)
    self.torque_des = np.zeros(3)

  def _step_angvel(self, t, dt, controller, dists=None, ts=None):
    if dists is None:
      dists = []

    state = self.rb.state()
    bodyz_force, angvel = controller.response(t=t, state=state)
    
    controlvars = {}
    if hasattr(controller, 'vars'):
      controlvars.update(controller.vars)

    if ts is not None:
      ts.add_point(time=t, **state.__dict__, force=bodyz_force, angvel=angvel, **controlvars)

    return self.step_angvel_raw(dt, bodyz_force, angvel, dists=dists)

  def step_angvel_raw(self, dt, bodyz_force, angvel, dists=None, linear_var=0.0, angular_var=0.0, latency=0, k=1, second_order=False, kw=1, kt=1, return_info=False):
    if dists is None:
      dists = []

    if latency > 0 or k < 1 or second_order:
      #print('dt', dt, latency, bodyz_force, angvel)
      #print('start', self.command_queue)

      for cmd in self.command_queue:
        self.command_queue[cmd] -= dt

      if len(self.command_queue) > 0:
        bodyz_force_des = self.bodyz_force_des
        angvel_des = self.angvel_des
        try:
          while self.command_queue.find_min(return_value=True)[1] <= 0:
            bodyz_force_des, angvel_des, _ =  self.command_queue.extract_min()
            angvel_des = np.array(angvel_des)
        except IndexError:
          pass
        #print('des', bodyz_force_des, angvel_des)
      else:
        bodyz_force_des = 0.0
        angvel_des = np.zeros(3)

      self.command_queue[(bodyz_force, tuple(angvel), self.cmd_id)] = latency
      #print('after', self.command_queue)
      self.cmd_id += 1

      self.bodyz_force_des = bodyz_force_des
      self.angvel_des = angvel_des

      self.curr_bodyz_force = self.curr_bodyz_force + k * (self.bodyz_force_des - self.curr_bodyz_force)

      if second_order:
        self.torque_des = kw * (self.angvel_des - self.curr_angvel) 
        self.curr_torque = self.curr_torque + kt * (self.torque_des - self.curr_torque)
        self.curr_angvel = self.curr_angvel + self.curr_torque
      else:
        self.curr_angvel = self.curr_angvel + k * (self.angvel_des - self.curr_angvel)

      #print('curr', self.curr_bodyz_force, self.curr_angvel)

      bodyz_force = self.curr_bodyz_force
      angvel = self.curr_angvel

    if bodyz_force < 0 or bodyz_force > self.force_limit:
      #print("Clipping force!", bodyz_force)
      bodyz_force = np.clip(bodyz_force, 0, self.force_limit)

    angvel_norm = np.linalg.norm(angvel)
    if angvel_norm > self.angvel_limit:
      print("Clipping angvel!", angvel)
      angvel *= self.angvel_limit / angvel_norm

    state = self.rb.state()
    force_world = bodyz_force * state.rot.apply(e3) + self.mass * self.gvec

    # Only force disturbances support now.
    for dist in dists:
      d = dist.get(state, (bodyz_force, 0))
      force_world += d[:3]

    self.rb.step_angvel(dt, force=force_world, angvel=angvel, linear_var=linear_var, angular_var=angular_var)

    if return_info:
      return state, self.bodyz_force_des, self.angvel_des, self.curr_bodyz_force, self.curr_angvel

    return state

  def step(self, t, dt, controller, **kwargs):
    step_f = self._step_angvel if controller.sim_angvel else self._step_torque
    return step_f(t, dt, controller, **kwargs)

  def _step_torque(self, t, dt, controller, dists=None, ts=None):
    if dists is None:
      dists = []

    state = self.rb.state()
    bodyz_force, torque = self.forcetorque_from_u(controller.response(t, state), dt=dt)

    if bodyz_force < 0 or bodyz_force > self.force_limit:
      #print("Clipping force!", bodyz_force)
      bodyz_force = np.clip(bodyz_force, 0, self.force_limit)

    torque_norm = np.linalg.norm(torque)
    if torque_norm > self.torque_limit:
      print("Clipping torque!", torque)
      torque *= self.torque_limit / torque_norm

    controlvars = {}
    if hasattr(controller, 'vars'):
      controlvars.update(controller.vars)

    if ts is not None:
      ts.add_point(time=t, **state.__dict__, force=bodyz_force, torque=torque, **controlvars)

    force_world = bodyz_force * state.rot.apply(e3) + self.mass * self.gvec
    torque_body = torque

    for dist in dists:
      d = dist.get(state, (bodyz_force, torque))
      force_world += d[:3]
      torque_body += d[3:]

    self.rb.step(dt, force=force_world, torque=torque_body)

  def simulate(self, dt, t_end, controller, dists=None):
    ts = TimeSeries()
    n_steps = int(round(t_end / dt)) - 1

    self.reset()
    for i in range(n_steps):
      self.step(i * dt, dt, controller, dists=dists, ts=ts)

      if self.vis:
        state = self.rb.state()
        self.vis.set_state(state.pos.copy(), state.rot)
        time.sleep(dt / self.vis_speed)

    ts.finalize()
    return ts

class QuadSimMotors(QuadSim):
  def __init__(self, model, **kwargs):
    super().__init__(model, **kwargs)
    self.reset()

  def reset(self):
    self.actrpm = self.model.rpm_from_forcetorque(self.model.mass * self.model.g, np.zeros(3))

  def forcetorque_from_u(self, desrpm, dt):
    alpha = self.model.motor_tc * dt
    self.actrpm = alpha * desrpm + (1 - alpha) * self.actrpm
    #self.actrpm = (alpha * desrpm[0] + (1 - alpha) * self.actrpm[0], alpha * desrpm[1] + (1 - alpha) * self.actrpm[1])

    #u = self.model.forcetorque_from_rpm(self.actrpm)
    #return u[0], self.model.forcetorque_from_rpm(desrpm)[1]
    return self.model.forcetorque_from_rpm(self.actrpm)
    #return self.actrpm
