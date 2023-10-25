import numpy as np

def filter_nan(ts, vs):
  good_vs = np.logical_not(np.isnan(vs))
  return ts[good_vs], vs[good_vs]

def first_that(f, arr):
  for i, v in enumerate(arr):
    if f(v):
      return i
  return None

def first_nonzero(arr):
  return first_that(lambda x: x, arr)

def first_greater_than(val, arr):
  return first_that(lambda x: x > val, arr)

def last_before(val, arr):
  """ Requires sorted (monotonic) arr """
  assert np.all(np.diff(arr) >= 0)
  return np.searchsorted(arr, val) - 1

def movmean(arr, n_avg):
  sums = np.cumsum(arr, dtype=float)
  sums[n_avg:] -= sums[:-n_avg]
  return sums[n_avg - 1 :] / n_avg
