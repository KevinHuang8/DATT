import os

import joblib

import numpy as np

from scipy.interpolate import interp1d
from scipy.signal import lfilter

cachedir = os.path.join(os.path.expanduser('~'), '.cache', 'python_utils')
memory = joblib.Memory(cachedir, verbose=0)

class DF1:
  def __init__(self, b, a, initial_value=0):
    assert a[0] == 1
    assert len(a) == len(b) == 3

    self.b = b
    self.a = a

    self.xd1 = self.xd2 = initial_value
    self.yd1 = self.yd2 = initial_value

  def filter(self, val):
    yn = self.b[0] * val + self.b[1] * self.xd1 + self.b[2] * self.xd2 - self.a[1] * self.yd1 - self.a[2] * self.yd2

    self.xd2 = self.xd1
    self.xd1 = val

    self.yd2 = self.yd1
    self.yd1 = yn

    return yn

@memory.cache
def exp_smooth(vals, alpha):
  """ y_{t+1} = (1 - alpha) y_t + alpha x_t
      TODO Use a scipy method? """
  smooth = np.array(vals[0], dtype=float)
  smoothed = [smooth.copy()]

  for i in range(1, len(vals)):
    smooth += -alpha * (smooth - vals[i])
    smoothed.append(smooth.copy())

  return np.array(smoothed)

def biquad_notch(freq, fs, Q):
  om = 2 * np.pi * freq / fs
  beta = np.tan(om / (2 * Q))

  n1 = 1 / (1 + beta)
  n2 = -2 * np.cos(om) / (1 + beta)
  n3 = (1 - beta) / (1 + beta)

  return [n1, n2, n1], [1, n2, n3]

@memory.cache
def dynamic_rpm_notch(times, rpmtimes, vals, rpms, fs, Q=5.0):
  rpm_at_val_times = interp1d(rpmtimes, rpms, fill_value="extrapolate", axis=0)(times)
  res = []
  filt = DF1([1, 1, 1], [1, 1, 1])
  for i, val in enumerate(vals):
    rpmnow = rpm_at_val_times[i]

    freq = rpmnow / 60.0
    if freq < fs / 2.0:
      filt.b, filt.a = biquad_notch(freq, fs, Q)

    else:
      print("WARNING: RPM too high for notch filtering! %d" % rpmnow)

    res.append(filt.filter(val))

  return np.array(res)

@memory.cache
def static_rpm_notch(vals, rpm, fs, Q=5.0):
  freq = rpm / 60.0
  if freq >= fs / 2.0:
    raise "ERROR: RPM %d too high for notch filtering" % rpm

  return lfilter(*biquad_notch(freq, fs, Q), vals, axis=0)

def pr_from_grav(grav):
  """ Returns pitch and roll from a matrix of gravity vectors in body frame """
  rolls = np.arctan2(grav[:, 1], grav[:, 2])
  pitches = np.arcsin(-grav[:, 0])
  return pitches, rolls

@memory.cache
def complementary_filter(weight, accs, gyros, dt, start_g=np.array((0, 0, 9.81))):
  assert 0 <= weight <= 1
  assert len(accs) == len(gyros)
  assert len(start_g) == len(accs[0]) == len(gyros[0]) == 3

  N = len(accs)

  ss = np.empty((N, 3))
  s = np.array(start_g)

  for i, (acc, gyro) in enumerate(zip(accs, gyros)):
    s_gyro = s + np.cross(s, gyro) * dt

    if np.linalg.norm(acc) < 1e-9:
      s_acc = s
      print("WARNING: Acc %d has low norm" % i, acc, gyro)
    else:
      s_acc = acc / np.linalg.norm(acc)

    s = weight * s_acc + (1 - weight) * s_gyro
    s /= np.linalg.norm(s)

    ss[i] = s

  return pr_from_grav(ss)

@memory.cache
def complementary_filter_bias(weight, weight_bias, accs, gyros, dt, start_g=np.array((0, 0, 9.81)), start_bias=np.zeros(3)):
  assert 0 <= weight <= 1
  assert weight_bias >= 0
  assert len(accs) == len(gyros)
  assert len(start_g) == len(start_bias) == len(accs[0]) == len(gyros[0]) == 3

  N = len(accs)

  ss = np.empty((N, 3))
  biases = np.empty((N, 3))
  s = np.array(start_g)
  bias = np.array(start_bias)

  for i, (acc, gyro) in enumerate(zip(accs, gyros)):
    s_gyro = s + np.cross(s, gyro - bias) * dt

    if np.linalg.norm(acc) < 1e-9:
      s_acc = s
      print("WARNING: Acc %d has low norm" % i, acc, gyro)
    else:
      s_acc = acc / np.linalg.norm(acc)

    s = weight * s_acc + (1 - weight) * s_gyro
    s /= np.linalg.norm(s)

    # Bias is driven by difference between acc estimate and gyro estimate
    bias += -weight_bias * np.cross(s_acc, s_gyro) * dt

    ss[i] = s
    biases[i] = bias.copy()

  pitches, rolls = pr_from_grav(ss)

  return pitches, rolls, biases
