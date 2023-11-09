import numpy as np
import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Quaternion routines adapted from rowan to use autograd
def qmultiply(q1, q2):
#   assert False
  return np.concatenate((
    np.array([q1[0] * q2[0]]), # w1w2
    q1[0] * q2[1:4] + q2[0] * q1[1:4] + np.cross(q1[1:4], q2[1:4])))

def qconjugate(q):
  return np.concatenate((q[0:1],-q[1:4]))

def qrotate(q, v):
  quat_v = np.concatenate((np.array([0]), v))
  return qmultiply(q, qmultiply(quat_v, qconjugate(q)))[1:]

def qexp(q):
  norm = np.linalg.norm(q[1:4])
  e = np.exp(q[0])
  result_w = e * np.cos(norm)
  if np.isclose(norm, 0):
    result_v = np.zeros(3)
  else:
    result_v = e * q[1:4] / norm * np.sin(norm)
  return np.concatenate((np.array([result_w]), result_v))

def qintegrate(q, v, dt, frame='body'):
	quat_v = np.concatenate((np.array([0]), v*dt/2))
	if frame == 'body':
		return qmultiply(q, qexp(quat_v))		
	if frame == 'world':
		return qmultiply(qexp(quat_v), q)

def qnormalize(q):
  return q / np.linalg.norm(q)

def qstandardize(q):
	if q[0] < 0:
		q *= -1
	return q / np.linalg.norm(q)

def qtoR(q):
  q0 = q[0]
  q1 = q[1]
  q2 = q[2]
  q3 = q[3]

  # First row of the rotation matrix
  r00 = 2 * (q0 * q0 + q1 * q1) - 1
  r01 = 2 * (q1 * q2 - q0 * q3)
  r02 = 2 * (q1 * q3 + q0 * q2)

  # Second row of the rotation matrix
  r10 = 2 * (q1 * q2 + q0 * q3)
  r11 = 2 * (q0 * q0 + q2 * q2) - 1
  r12 = 2 * (q2 * q3 - q0 * q1)

  # Third row of the rotation matrix
  r20 = 2 * (q1 * q3 - q0 * q2)
  r21 = 2 * (q2 * q3 + q0 * q1)
  r22 = 2 * (q0 * q0 + q3 * q3) - 1

  # 3x3 rotation matrix
  rot_matrix = np.array([[r00, r01, r02], \
                         [r10, r11, r12], \
                         [r20, r21, r22]])

  return rot_matrix

# torch versions:
def sqaured_distance_torch(x, x_d):
  '''
  x: actual state, tensor, (N, m)
  x_d: desired, tensor, (m,)
  output: squared distance, tensor, (N,)
  '''
  return torch.einsum('ij,ij->i', x-x_d, x-x_d)

def qdistance_torch(q1, q2):
  '''
  q1: tensor, (N, 4)
  q2: tensor, (N, 4)
  output: tensor, (N,)
  distance = 1 - <q1,q2>^2
  '''
  return 1 - torch.einsum('...i, ...i -> ...', q1, q2)**2

def qmultiply_torch(q1, q2):
  '''
  q1: tensor, (N, 4)
  q2: tensor, (N, 4)
  output: tensor, (N, 4)
  '''
  temp = torch.zeros_like(q1)
  v1 = q1[..., 1:]
  v2 = q2[..., 1:]
  temp[..., 0] = q1[..., 0] * q2[..., 0] - torch.einsum('...i, ...i -> ...', v1, v2)
  temp[..., 1:] = q1[..., 0:1] * v2 + q2[..., 0:1] * v1 + torch.cross(v1, v2, dim=1)
  return temp

def qconjugate_torch(q):
  temp = torch.zeros_like(q)
  temp[:,0] = q[:,0]
  temp[:,1:] = -q[:,1:]
  return temp

def z_from_q(q):
  '''
  q: (N, H, 4)
  output: (N, H, 3)
  '''
  e3 = torch.tensor((0, 0, 1.0)).view(1, 1, 3)
  temp = 2. * torch.cross(q[:, :, 1:], e3)
  return e3 + q[:, :, 0:1] * temp + torch.cross(q[:, :, 1:], temp)

def qrotate_torch(q, v):
  '''
  q: tensor, (N, 4)
  v: tensor, (N, 3)
  output: tensor, (N, 3)
  '''
  # a more efficient way suggested by Alex:
  temp = 2. * torch.cross(q[:, 1:], v)
  return v + q[:, 0:1] * temp + torch.cross(q[:, 1:], temp)

def qexp_torch(q):
  '''
  q: tensor, (N, 4)
  output: tensor, (N, 4)
  '''
  norm = torch.linalg.norm(q[:,1:], dim=1)
  e = torch.exp(q[:,0])
  result_w = e * torch.cos(norm)

  N = q.shape[0]
  result_v = e.view(N,1) * q[:,1:] / norm.view(N,1) * torch.sin(norm).view(N,1)
  result_v[torch.isnan(result_v)] = 0

  return torch.cat((result_w.view(N,1), result_v), dim=1)

def qintegrate_torch(q, v, dt, frame='body'):
  '''
  q: tensor, (N, 4)
  v: tensor, (N, 3)
  output: tensor, (N, 4)
  '''
  quat_v = torch.zeros_like(q)
  quat_v[:,1:] = v * dt / 2.
  if frame == 'body':
    return qmultiply_torch(q, qexp_torch(quat_v))
  if frame == 'world':
    return qmultiply_torch(qexp_torch(quat_v), q)

def qstandardize_torch(q):
  '''
  q: tensor, (N, 4)
  output: tensor, (N, 4)
  '''
  return torch.where(q[:, 0:1] < 0, -q, q)


def qtoR_torch(q):
  '''
  q: tensor, (N, 4)
  output: rotation matrix tensor, (N, 3, 3)
  '''
  q0 = q[:, 0]
  q1 = q[:, 1]
  q2 = q[:, 2]
  q3 = q[:, 3]
  R = torch.zeros(q.shape[0], 3, 3)

  # First row of the rotation matrix
  R[:, 0, 0] = 2 * (q0 * q0 + q1 * q1) - 1
  R[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
  R[:, 0, 2] = 2 * (q1 * q3 + q0 * q2)

  # Second row of the rotation matrix
  R[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
  R[:, 1, 1] = 2 * (q0 * q0 + q2 * q2) - 1
  R[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)

  # Third row of the rotation matrix
  R[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
  R[:, 2, 1] = 2 * (q2 * q3 + q0 * q1)
  R[:, 2, 2] = 2 * (q0 * q0 + q3 * q3) - 1

  return R