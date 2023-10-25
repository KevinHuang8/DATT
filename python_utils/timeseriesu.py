import numpy as np

def apply_f_to_ts(src, f):
  for k, data in src.__dict__.items():
    if isinstance(data, TimeSeries):
      src[k] = f(data)
      setattr(src, k, src[k])

    elif isinstance(data, DataSet):
      apply_f_to_ts(data, f)

def trans_data(src, dest, f):
  # Ideally, this meta data should not be in the same namespace as the data.
  forbidden_keys = ["times", "meta_times", "metadata", "finalized", "t0"]
  if not hasattr(src, '__dict__'):
    if type(src) is dict:
      for k, data in src.items():
        dest[k] = f(data)
    else:
      # What to do
      assert False

  else:
    for k, data in src.__dict__.items():
      if k in forbidden_keys:
        continue

      if isinstance(data, dict):
        dest[k] = BasicAttrDict()
        trans_data(data, dest[k], f)
      else:
        dest[k] = f(data)

      setattr(dest, k, dest[k])

def f_retimed(ts, newts, **kwargs):
  from scipy.interpolate import interp1d
  from scipy.spatial.transform import Rotation as R

  def f(data, ts=ts, newts=newts):
    if len(data) and type(data[0]) == type(R.identity()):
      from scipy.spatial.transform import Slerp
      rots = R.from_matrix([rot.as_matrix() for rot in data])
      return Slerp(ts, rots)(newts)

    return interp1d(ts, data, axis=0, **kwargs)(newts)

  return f

def f_masked(mask):
  def f(data, mask=mask):
    # Only to deal with numpy Rotation bug
    if isinstance(data, list):
      if len(data) == len(mask):
        new_list = []
        for i, x in enumerate(data):
          if mask[i]:
            new_list.append(x)
        return new_list
      return data

    # For any auxiliary vars that may have been added...
    if isinstance(data, (str, int, float, np.generic)):
      return data

    return data[mask]

  return f

def retimed_copy(src, dest, oldts, newts, **kwargs):
  return trans_data(src, dest, f_retimed(oldts, newts, **kwargs))

def masked_copy(src, dest, mask):
  return trans_data(src, dest, f_masked(mask))

class BasicAttrDict(dict):
  pass

class DataSet(dict):
  def _item_map(self, f):
    ret = DataSet()
    for k, v in self.items():
      ret[k] = f(v)
      setattr(ret, k, ret[k])

    return ret

  def add_point(self, key, delim='/', ts_metadata=None, **data):
    key = key.strip(delim)

    del_ind = key.find(delim)
    if del_ind != -1:
      fkey = key[:del_ind]
      if fkey not in self:
        self[fkey] = DataSet()
        setattr(self, fkey, self[fkey])

      self[fkey].add_point(key[del_ind + 1:], delim=delim, ts_metadata=ts_metadata, **data)

    else:
      if key not in self:
        self[key] = TimeSeries(metadata=ts_metadata)
        setattr(self, key, self[key])

      self[key].add_point(**data)

  def method_map(self, method_name, *args):
    return self._item_map(lambda obj, args=args: getattr(obj, method_name)(*args))

  def get_view(self, start_time, end_time):
    return self.method_map('get_view', start_time, end_time)

  def get_after(self, start_time):
    return self.method_map('get_after', start_time)

  def get_before(self, end_time):
    return self.method_map('get_before', end_time)

  def get_multiview(self, mask):
    return self.method_map('get_multiview', mask)

  def finalize(self):
    [v.finalize() for v in self.values()]

class TimeSeries(dict):
  def __init__(self, metadata=None):
    self.times = []
    self.meta_times = []
    self.metadata = metadata
    self.finalized = False

  def __len__(self):
    return len(self.times)

  def _build_dict(self, d, obj, ind):
    for name, val in d.items():
      if isinstance(val, dict):
        obj[name] = BasicAttrDict()
        self._build_dict(val, obj[name], ind)
      elif isinstance(val, np.ndarray) or isinstance(val, list):
        obj[name] = val[ind]
      else:
        print(val)
        assert False

      setattr(obj, name, obj[name])

  def point_iter(self):
    assert self.finalized
    for i in range(len(self.times)):
      obj = BasicAttrDict()
      self._build_dict(self, obj, i)
      obj.t = self.times[i]
      if len(self.meta_times):
        obj.meta_t = self.meta_times[i]

      yield obj

  def sub_add(self, d, **kwargs):
    for name, val in kwargs.items():
      first = name not in d
      if isinstance(val, dict):
        if first:
          d[name] = BasicAttrDict()

        self.sub_add(d[name], **val)

      else:
        if first:
          d[name] = []

        d[name].append(val)

      if first:
        setattr(d, name, d[name])

  def add_point(self, time, meta_time=None, **kwargs):
    assert not self.finalized

    self.times.append(time)

    if meta_time is not None:
      self.meta_times.append(meta_time)

    self.sub_add(self, **kwargs)

  def _finalize(self, name, vals, d):
    test = np.array(vals)
    # Deal with scipy Rotation object bug
    if len(test.shape) == 32:
      d[name] = vals
    else:
      d[name] = test

    setattr(d, name, d[name])

  def finalize(self):
    self.times = np.array(self.times)
    self.meta_times = np.array(self.meta_times)
    self.apply_f(self._finalize, self)

    if len(self.times):
      self.t0 = self.times[0]
      self.finalized = True

  def apply_f(self, f, d, *args, **kwargs):
    for name, vals in d.items():
      if isinstance(vals, dict):
        self.apply_f(f, vals, *args, **kwargs)
      else:
        f(name, vals, d, *args, **kwargs)

  def _delete_inds(self, name, vals, d, inds):
    # Only to deal with numpy Rotation bug
    if isinstance(vals, list):
      d[name] = [val for i, val in enumerate(vals) if i not in inds]
    else:
      d[name] = np.delete(d[name], inds, axis=0)

    setattr(d, name, d[name])

  def remove_dup_times(self):
    assert self.finalized

    timedups = np.hstack((np.diff(self.times) == 0, False))
    self.times = np.delete(self.times, timedups)
    if self.meta_times:
      self.meta_times = np.delete(self.meta_times, timedups)
    self.apply_f(self._delete_inds, self, timedups)

  def get_masked_view(self, timemask):
    assert self.finalized

    ret = TimeSeries(metadata=self.metadata)
    ret.finalized = True
    ret.times = self.times[timemask].copy()
    if hasattr(self, 't0'):
      if len(ret.times):
        ret.t0 = ret.times[0]
      else:
        ret.t0 = self.t0
    if len(self.meta_times):
      ret.meta_times = self.meta_times[timemask].copy()
    else:
      ret.meta_times = self.meta_times

    masked_copy(self, ret, timemask)

    return ret

  def get_multiview(self, timess):
    mask = np.zeros(len(self.times), dtype=bool)
    for times in timess:
      mask = np.logical_or(mask, np.logical_and(times[0] <= self.times, self.times <= times[1]))
    return self.get_masked_view(mask)

  def get_view(self, start_time, end_time):
    return self.get_multiview([(start_time, end_time)])

  def get_after(self, start_time):
    return self.get_view(start_time, np.inf)

  def get_before(self, end_time):
    return self.get_view(-np.inf, end_time)

  def get_all(self):
    return self.get_view(-np.inf, np.inf)

  def retime(self, fieldstr, newts, fill_value=0.0, **kwargs):
    from scipy.interpolate import interp1d

    fieldlist = fieldstr.split('.')
    obj = self
    for field in fieldlist:
      obj = getattr(obj, field)

    return interp1d(self.times, obj, axis=0, bounds_error=False, fill_value=fill_value, **kwargs)(newts)

  def retimeall(self, newts, **kwargs):
    assert np.all(np.diff(newts) >= 0)
    assert np.all(np.diff(self.times) >= 0)


    newts = newts[np.logical_and(newts >= self.times[0], newts <= self.times[-1])]

    ret = TimeSeries(metadata=self.metadata)
    ret.finalized = True
    ret.times = newts

    if hasattr(self, 't0'):
      if len(ret.times):
        ret.t0 = ret.times[0]
      else:
        ret.t0 = self.t0

    ret.meta_times = None

    retimed_copy(self, ret, self.times, newts, **kwargs)

    return ret
