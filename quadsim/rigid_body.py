import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from DATT.python_utils.mathu import quat_mult, vector_quat, normalized
from DATT.python_utils.rigid_body import euler_int, so3_quat_int

# class State:
#   def __init__(self, pos=np.zeros(3), vel=np.zeros(3), rot=R.identity(), ang=np.zeros(3)):
#     self.pos = pos # R^3
#     self.vel = vel # R^3
#     self.rot = rot # Scipy Rotation rot.as_matrix() rot.as_quat()
#     self.ang = ang # R^3

class State_struct:
  def __init__(self, pos=np.zeros(3), 
                     vel=np.zeros(3),
                     acc = np.zeros(3),
                     jerk = np.zeros(3), 
                     snap = np.zeros(3),
                     rot=R.from_quat(np.array([0.,0.,0.,1.])), 
                     ang=np.zeros(3)):
    
    self.pos = pos # R^3
    self.vel = vel # R^3
    self.acc = acc
    self.jerk = jerk
    self.snap = snap
    self.rot = rot # Scipy Rotation rot.as_matrix() rot.as_quat()
    self.ang = ang # R^3
    self.t = 0.
  
  def get_vec_state_numpy(self, q_order = 'xyzw', ):

    if q_order=='xyzw':
      return np.r_[self.pos, self.vel, self.rot.as_quat(), self.ang]
    else:
      #quaternion -> w,x,y,z
      return np.r_[self.pos, self.vel, np.roll(self.rot.as_quat(), 1), self.ang]
  def get_vec_state_torch(self, q_order = 'xyzw'):
    return torch.tensor(self.get_vec_state_numpy(q_order=q_order))

  def update_from_vec(self, state_vec):
    self.pos = state_vec[:3] # R^3
    self.vel = state_vec[3:6] # R^3
    self.rot = R.from_quat(state_vec[6:10])
    self.ang = state_vec[10:]

class RigidBody:
  def __init__(self, mass=1, inertia=np.eye(3)):
    self.mass = mass
    self.I = inertia
    self.Iinv = np.linalg.inv(self.I)
    self.setstate(State_struct())

  def setstate(self, state : State_struct):
    """
        pos and vel are in the fixed frame
        rot transforms from the body frame to the fixed frame.
        ang is in the body frame
    """
    self.pos = state.pos.copy()
    self.vel = state.vel.copy()
    quat_wlast = state.rot.as_quat()
    self.quat = np.hstack((quat_wlast[3], quat_wlast[0:3]))
    self.ang = state.ang.copy()

  def step(self, dt, force, torque):
    """
        force is in the fixed frame
        torque is in the body frame
    """
    accel = force / self.mass
    # Euler equation:
    # Torque = I alpha + om x (I om)
    alpha = self.Iinv.dot(torque - np.cross(self.ang, self.I.dot(self.ang)))

    self.pos += euler_int(dt, self.vel, accel)
    self.vel += accel * dt

    dang = euler_int(dt, self.ang, alpha)
    self.quat = so3_quat_int(self.quat, dang, dang_in_body=True)
    self.ang += alpha * dt

  def step_angvel(self, dt, force, angvel, linear_var=0.0, angular_var=0.0):
    """
        force is in the fixed frame
        angvel is in the body frame
    """
    accel = force / self.mass

    self.pos += euler_int(dt, self.vel, accel)
    self.vel += accel * dt + np.random.normal(loc=0.0, scale=linear_var, size=3)

    dang = euler_int(dt, angvel, 0)
    dang = dang + np.random.normal(loc=0.0, scale=angular_var, size=3)
    self.quat = so3_quat_int(self.quat, dang, dang_in_body=True)
    self.ang = angvel.copy()

  def getpos(self):
    return self.pos.copy()
  def getvel(self):
    return self.vel.copy()
  def getrot(self):
    return R.from_quat(np.hstack((self.quat[1:4], self.quat[0])))
  def getang(self):
    return self.ang.copy()

  def state(self):
    return State_struct(pos=self.getpos(), 
                        vel=self.getvel(), 
                        rot=self.getrot(), 
                        ang=self.getang())
