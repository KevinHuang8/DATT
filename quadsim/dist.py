import numpy as np

from python_utils.wind import WindModel

class ForceDisturbance:
  def apply(self, model):
    pass

  def get(self, state, control):
    return np.hstack((self.force(state, control), np.zeros(3)))

class TorqueDisturbance:
  def apply(self, model):
    pass

  def get(self, state, control):
    return np.hstack((np.zeros(3), self.torque(state, control)))

class ModelDisturbance:
  def get(self, state, control):
    return np.zeros(6)

class LinearDrag(ForceDisturbance):
  def __init__(self, scale):
    self.c = scale

  def force(self, state, control):
    return -self.c * state.vel

class ConstantForce(ForceDisturbance):
  def __init__(self, scale):
    self.c = scale

  def force(self, state, control):
    return self.c

class WindField(ForceDisturbance):
  def __init__(self, pos, direction, vmax=10.0, noisevar=0.5, decay_lat=4, decay_long=0.6):
    self.vmax = vmax
    self.model = WindModel(pos, direction, vmax=self.vmax, radius=0.2, decay_lat=decay_lat, decay_long=decay_long, dispangle=np.radians(15))
    self.noisevar = noisevar

  def force(self, state, control):
    windvel = self.model.velocity(state.pos)
    noise_scale = np.linalg.norm(windvel) / self.vmax
    return 0.2 * windvel + self.noisevar * noise_scale * np.random.normal(size=3)

class MassDisturbance(ModelDisturbance):
  def __init__(self, scale):
    self.c = scale

  def apply(self, model):
    model.mass *= self.c

class InertiaDisturbance(ModelDisturbance):
  """ Assume diagonal inertia """
  def __init__(self, scales):
    self.scales = np.array(scales)

  def apply(self, model):
    assert np.allclose(model.I, np.diag(np.diag(model.I)))

    model.I[0] *= self.scales[0]
    model.I[1] *= self.scales[1]
    model.I[2] *= self.scales[2]

class MotorModelDisturbance(ModelDisturbance):
  def __init__(self, scale):
    self.c = scale

  def apply(self, model):
    model.params['motor_thrust_coeffs'] = self.c * np.array(model.params['motor_thrust_coeffs'])
    model.update()
