import contextlib
import sys

class Dummy:
  def write(self, _):
    pass

@contextlib.contextmanager
def nostdout():
  save = sys.stdout
  sys.stdout = Dummy()
  yield
  sys.stdout = save

@contextlib.contextmanager
def nostderr():
  save = sys.stderr
  sys.stderr = Dummy()
  yield
  sys.stderr = save
