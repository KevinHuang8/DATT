import numpy as np
import time

class Timer():
  '''
  for code profiling
  '''
  def __init__(self, topics):
    self.t = 0.
    self.stats = {}
    for topic in topics:
      self.stats[topic] = []

  def tic(self):
    self.t = time.time()

  def toc(self, topic):
    self.stats[topic].append(time.time() - self.t)