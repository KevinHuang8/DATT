import numpy as np

from python_utils.quadrotoru import Quadrotor

class RBModel:
  def __init__(self, mass, I, g):
    self.mass = mass
    self.I = I
    self.g = g

class IdentityModel(RBModel):
  def __init__(self):
    super().__init__(1, np.eye(3), general_params()['g'])

class QuadModel(RBModel):
  def __init__(self, params):
    assert type(params) is dict
    self.params = params
    self.update()

  def update(self):
    quadrotor = Quadrotor(**self.params)
    super().__init__(quadrotor.mass, quadrotor.I, quadrotor.g)

    self.mixer_inv = quadrotor.mixer_inv
    self.mixer = quadrotor.mixer
    self.motor_thrust = quadrotor.motor_thrust
    self.motor_tc = quadrotor.motor_tc

    self.reset()

  def reset(self):
    self.started = False

  def rpm_from_forcetorque(self, force, torque):
    rotorforces = self.mixer_inv.dot(np.hstack((force, torque)))
    rotorforces = np.clip(rotorforces, 0.0, np.inf)
    ret = np.real(np.array([np.max((self.motor_thrust - rf).roots) for rf in rotorforces]))
    return ret

  def forcetorque_from_rpm(self, rpm):
    u = self.mixer.dot(self.motor_thrust(rpm))
    return u[0], u[1:]

def general_params():
  return dict(g=9.8)

def rocky09_params():
  return dict(
    mass=0.665,
    inertia=np.diag((0.005, 0.007, 0.006)),
    motor_thrust_coeffs=[5.70556841e-08, -1.22857889e-04, 6.33954558e-02],
    motor_torque_scale=0.015,
    center_of_mass=np.array((0.0, 0.0, 0.0)),
    motor_inertia=0.0,
    motor_arm_length=0.153,
    motor_spread_angle=0.8798653713475464,
    motor_tc=30,
  )

def rocky09():
  return QuadModel({ **rocky09_params(), **general_params() })
