# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD-3-Clause


from .__about__ import (
    __version__,
    __license__,
)

from .network import network
from .dataset import dataset

# this is for "from <package_name> import *"
__all__ = ["network","dataset",]
