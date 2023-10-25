import numpy as np

def deriv_fitting_matrix(degree, t_end=1.0):
  """
    Returns A s.t. that the vector x that satisfies

    Ax = b

    contains polynomial coefficients

      p(t) = x_1 + x_2 t + x_3 t^2 ...

    so that

    p(0) = b_1
    p^(i)(0) = b_{i+1}
    p(t_end) = b_{degree / 2}
    p^(i)(t_end) = b_{degree / 2 + i + 1}

    i.e. the first degree / 2 derivatives of p at 0 match the first degree / 2
         entries of b and the first degree / 2 derivatives of p at 1, match the last
         degree / 2 entries of b
  """

  assert degree % 2 == 0

  A = np.zeros((degree, degree))

  ts = t_end ** np.array(range(degree))

  constant_term = 1
  poly = np.ones(degree)
  for i in range(degree // 2):
    A[i, i] = constant_term
    A[degree // 2 + i, :] = np.hstack((np.zeros(i), poly * ts[:degree - i]))
    poly = np.polynomial.polynomial.polyder(poly)
    constant_term *= (i + 1)

  return A

if __name__ == "__main__":
  for i in range(1, 5):
    mat = deriv_fitting_matrix(2 * i)
    print(mat)
    print(np.linalg.inv(mat))
