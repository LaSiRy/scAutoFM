# Ensure NumPy 2 shims load before timm (via supernet) in hologenomnics.
from lib import numpy_compat  # noqa: F401

from .supernet import *
