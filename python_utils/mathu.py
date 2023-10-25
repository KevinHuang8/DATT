import numpy as np

R_slice = (slice(3), slice(3))
t_slice = (slice(3), 3)

def e(n, i):
  ret = np.zeros(n)
  ret[i - 1] = 1.0
  return ret

e1 = e(3, 1)
e2 = e(3, 2)
e3 = e(3, 3)

def get_tf(R, t):
  tf = np.zeros((4, 4))
  tf[R_slice] = R
  tf[t_slice] = t
  tf[3, 3] = 1
  assert_valid_tf(tf)
  return tf

def assert_valid_tf(tf):
  assert tf.shape == (4, 4), "TF is not 4 x 4"
  assert tf[3, 3] == 1, "tf[3, 3] is not 1"
  assert all(tf[3, :3] == 0), "tf[3, :3] is not all zeros"
  R = tf[R_slice]
  assert np.allclose(R.T.dot(R), np.eye(3), atol=1e-6), "tf R is not in SO(3)"

def invert_tf(tf):
  assert_valid_tf(tf)
  R = tf[R_slice]
  return get_tf(R.T, -R.T.dot(tf[t_slice]))

def R_x(angle):
  """ angle is in radians. """
  return np.array((
    (1, 0, 0),
    (0, np.cos(angle), -np.sin(angle)),
    (0, np.sin(angle), np.cos(angle))
  ))

def R_y(angle):
  """ angle is in radians. """
  return np.array((
    (np.cos(angle), 0, np.sin(angle)),
    (0, 1, 0),
    (-np.sin(angle), 0, np.cos(angle))
  ))

def R_z(angle):
  """ angle is in radians. """
  return np.array((
    (np.cos(angle), -np.sin(angle), 0),
    (np.sin(angle), np.cos(angle), 0),
    (0, 0, 1)
  ))

def euler_matrix_extrinsic_zyx(yaw, pitch, roll):
  """ angles are in radians. """
  return R_x(roll).dot(R_y(pitch)).dot(R_z(yaw))

def euler_matrix_intrinsic_zyx(yaw, pitch, roll):
  """ angles are in radians. """
  return R_z(yaw).dot(R_y(pitch)).dot(R_x(roll))

def quat_mult(a, b):
  return np.array((
    a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
    a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
    a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
    a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
  ))

def quat_rotate(quat, vec):
  t = 2 * np.cross(quat[1:], vec)
  return vec + quat[0] * t + np.cross(quat[1:], t)

def quat_inverse(quat):
  quat_inv = quat.copy()
  quat_inv[0] = -quat_inv[0]
  return quat_inv

def matrix_from_quat(q):
  qw, qx, qy, qz = q
  return np.array((
    (1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw),
    (2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw),
    (2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2)
  ))

def quat_from_axis_angle(axis, angle):
  assert np.isclose(np.linalg.norm(axis), 1)
  return np.hstack(((np.cos(angle / 2),), np.sin(angle / 2) * axis))

def quat_identity():
  return np.array((1., 0., 0., 0.))

def vector_quat(v):
  return np.array((0, v[0], v[1], v[2]))

def skew_matrix(v):
  return np.array(((0, -v[2], v[1]),
                   (v[2], 0, -v[0]),
                   (-v[1], v[0], 0)))

def hat(v):
  return skew_matrix(v)

def vee(m):
  assert np.allclose(m.T, -m)
  return np.array((-m[1, 2], m[0, 2], -m[0, 1]))

def axis_from_quat(q):
  return q[1:] / np.linalg.norm(q[1:])

def normang(ang):
  """
    ang is in radians
    returns in [-pi, pi]
  """
  return (ang + np.pi) % (2 * np.pi) - np.pi

def normangpos(ang):
  """ ang is in radians
      returns in [0, 2pi]
  """
  return ang % (2 * np.pi)

def smoothang(ang):
  """
    "unnormalizes" such that angle value is smooth and measures cumulative angle
  """
  return np.hstack((ang[0], ang[0] + np.cumsum(normang(np.diff(ang)))))

def avgang(angs):
  """ Angles in radians. Returns "average" angle using unit vector method """
  return np.arctan2(np.sum(np.sin(angs)), np.sum(np.cos(angs)))

def rodmat(k):
  ang = np.linalg.norm(k)
  if ang < 1e-9:
    return np.eye(3)

  K = hat(k / ang)
  return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K.dot(K)

def rodrot(k, v):
  ang = np.linalg.norm(k)
  ax = k / ang

  c = np.cos(ang)
  s = np.sin(ang)
  return v * c + np.cross(ax, v) * s + ax.dot(v) * (1 - c) * ax

def rot_from_z_yaw_zyx(z, yaw):
  """ Yaw as defined via Euler angles ZYX """
  c1 = np.array((-np.sin(yaw), np.cos(yaw), np.zeros_like(yaw))).T
  x = np.cross(c1, z)
  x /= np.linalg.norm(x, axis=-1)[:, np.newaxis]
  y = np.cross(z, x)

  return np.stack((x, y, z), axis=-1)

def rot_from_z_yaw_zxy(z, yaw):
  """ Yaw as defined via Euler angles ZXY """
  c1 = np.array((np.cos(yaw), np.sin(yaw), np.zeros_like(yaw))).T
  y = np.cross(z, c1)
  y /= np.linalg.norm(y, axis=-1)[:, np.newaxis]
  x = np.cross(y, z)

  return np.stack((x, y, z), axis=-1)

def normalized(v):
  return v / np.linalg.norm(v)

def numerical_jacobian(f, xs, dx=1e-6):
  """
      f is a function that accepts input of shape (n_points, input_dim)
      and outputs (n_points, output_dim)

      return the jacobian as (n_points, output_dim, input_dim)
  """
  if len(xs.shape) == 1:
    xs = xs[np.newaxis, :]

  assert len(xs.shape) == 2

  ys = f(xs)

  x_dim = xs.shape[1]
  y_dim = ys.shape[1]

  jac = np.empty((xs.shape[0], y_dim, x_dim))

  for i in range(x_dim):
    x_try = xs + dx * e(x_dim, i + 1)
    jac[:, :, i] = (f(x_try) - ys) / dx

  return jac

def numerical_grad_mani(f, x, x_dim, f_addhat, dx=1e-6):
  """
      Assuming f outputs elements in a Euclidean space,
      but input x may be on a manifold.
      f_addhat(x, dx) = x + hat(dx), where + is addition on the manifold.
      hat(dx) should project from the lie algebra to the manifold
  """
  y = f(x)

  assert len(y.shape) <= 1
  y_dim = y.shape[0] if len(y.shape) == 1 else 1

  grad = np.empty((y_dim, x_dim))
  for i in range(x_dim):
    Dx = dx * e(x_dim, i + 1)
    grad[:, i] = (f(f_addhat(x, Dx)) - y) / dx

  if grad.shape[0] == 1:
    return grad[0]

  return grad

def numerical_hess_mani(f, x, x_dim, f_addhat, dx=1e-6):
  def gradf(x):
    return numerical_grad_mani(f, x, x_dim, f_addhat, dx=dx)

  return numerical_grad_mani(gradf, x, x_dim, f_addhat, dx=dx)

def gradient_descent_mani(f, x, f_addhat, alpha, maxiter=10000, min_cost_change=1e-10, min_grad_norm=1e-6, print_progress=False, debug_grad=False, show_hessian=False):
  """
     f(x) = (cost, gradient)
  """
  prev_cost = 99e99
  for i in range(maxiter):
    cost, grad = f(x)

    if print_progress and i and not i % 100:
      print(i, cost, x)

    if debug_grad:
      numgrad = numerical_grad_mani(lambda x: f(x)[0], x, len(grad), f_addhat, dx=1e-8)
      assert np.allclose(grad, numgrad)

    if np.linalg.norm(grad) < min_grad_norm:
      if print_progress:
        print("Gradient small, exiting.")
      break

    if abs(cost - prev_cost) < min_cost_change:
      if print_progress:
        print("Small change in cost, exiting.")
      break

    prev_cost = prev_cost

    x = f_addhat(x, -alpha * grad)

  else:
    print("WARNING: maxiter = %d reached, exiting." % maxiter)

  if show_hessian:
    numhess = numerical_hess_mani(lambda x: f(x)[0], x, len(grad), f_addhat, dx=1e-5)
    vals, vecs = np.linalg.eigh(numhess)
    print("Hessian eigenvalues:", vals)
    for i in range(len(vecs)):
      print(vals[i], vecs[:, i])

  return x, i
