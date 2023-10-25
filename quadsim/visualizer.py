import pathlib

import numpy as np

try:
  import meshcat
  import meshcat.geometry as g
  import meshcat.transformations as tf
  MESHCAT_FOUND = True
except ImportError:
  print("WARNING: meshcat not installed. no vis supported.")
  MESHCAT_FOUND = False

class Vis:
  def __init__(self):
    if not MESHCAT_FOUND:
      return

    self.vis = meshcat.Visualizer()
    self.vis.open()
    self.vis["/Cameras/default"].set_transform(
    tf.translation_matrix([0,0,0]).dot(tf.euler_matrix(0, np.radians(30), 0)))

    self.vis["/Cameras/default/rotated/<object>"].set_transform(tf.translation_matrix([-1, 0, 0]))

    meshpath = pathlib.Path(__file__).parent / "mesh" / "quad.stl"
    self.vis["Quadrotor"].set_object(g.StlMeshGeometry.from_file(meshpath))

  def set_state(self, pos, rot):
    if not MESHCAT_FOUND:
      return

    quat = rot.as_quat()
    quat = [quat[3], quat[0], quat[1], quat[2]]

    self.vis["Quadrotor"].set_transform(tf.translation_matrix(pos).dot(tf.quaternion_matrix(quat).dot(tf.euler_matrix(0, 0, np.pi))))
