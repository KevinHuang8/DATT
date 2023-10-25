import time

class Timed:
  """ This fails if methods are accessed but not called! """
  def __getattribute__(self, name):
    if name in ['_time_dict', '__getattr__']:
      return object.__getattribute__(self, name)

    if not hasattr(self, '_time_dict'):
      self._time_dict = {}

    d = self._time_dict

    if callable(object.__getattribute__(self, name)):
      now = time.time()
      if name in d:
        self.dt = now - d[name]
      else:
        self.dt = None

      d[name] = now

    return object.__getattribute__(self, name)

if __name__ == "__main__":
  class Test(Timed):
    def __init__(self):
      self.x = 0

    def inc(self):
      self.x += 1
      print("inc", self.dt)

    def dec(self):
      self.x -= 1
      print("dec", self.dt)

  t = Test()
  t.dec()
  t.inc()
  t.inc()
  time.sleep(0.5)
  t.inc()
  time.sleep(0.2)
  t.inc()
  print(t.x)
  t.dec()
