# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation
import warnings

def axis_angle_to_euler(axis, angle):
    """
    Convert axis-angle to Euler angles using zyz convention
    """ 
    R = Rotation.from_rotvec(np.array(axis) * angle)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return R.as_euler('zyz')

def axis_angle_to_rotmat(axis, angle):
    """
    Convert axis-angle to rotation matrix
    """
    R = (1 - np.cos(angle)) * np.outer(axis,axis) + \
        np.cos(angle) * np.identity(3) + \
        np.sin(angle) * np.cross(axis, np.identity(3) * -1)
    
    return R

