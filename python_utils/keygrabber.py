"""
KeyGrabber

Reads key by key keyboard input from the commandline.
Call read to get a list of characters typed.
Restores terminal settings on program exit.
"""

import atexit
import sys
import termios
import tty

class KeyGrabber:
  def __init__(self):
    self._setup_keys()
    atexit.register(self._restore_keys)

  def _setup_keys(self):
    fd = sys.stdin.fileno()

    self.old_settings = termios.tcgetattr(fd)

    tty.setcbreak(fd)
    new_settings = termios.tcgetattr(fd)
    new_settings[6][termios.VMIN] = 0
    new_settings[6][termios.VTIME] = 0

    termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)

  def _restore_keys(self):
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old_settings)

  def read(self):
    chars = []
    while c := sys.stdin.read(1):
      chars.append(c)
    return chars

if __name__ == "__main__":
  import time

  kg = KeyGrabber()
  while 1:
    chars = kg.read()
    if chars:
      print(f"Got {len(chars)} chars: ", chars)
    time.sleep(0.02)
