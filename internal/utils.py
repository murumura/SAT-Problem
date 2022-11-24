"""Utility functions."""
import enum
import os
from typing import Any, Dict, Optional, Union
import numpy as np
from PIL import Image

class DataExtension(enum.Enum):
  """Dataset split."""
  TEXT = 'txt'
  DIMACS = 'dimacs'

def open_file(pth, mode='r'):
  return open(pth, mode=mode)


def file_exists(pth):
  return os.path.exists(pth)


def listdir(pth):
  return os.listdir(pth)


def isdir(pth):
  return os.path.isdir(pth)


def makedirs(pth):
  if not file_exists(pth):
    os.makedirs(pth)

def load_img(pth: str) -> np.ndarray:
  """Load an image and cast to float32."""
  with open_file(pth, 'rb') as f:
    image = np.array(Image.open(f), dtype=np.float32)
  return image