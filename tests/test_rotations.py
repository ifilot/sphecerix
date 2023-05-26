import unittest
import numpy as np
import sys
import os
from scipy.spatial.transform import Rotation

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import functions
from sphecerix import axis_angle_to_rotmat, axis_angle_to_euler

class TestRotations(unittest.TestCase):
    """
    Test the construction of rotation matrices from axis-angle
    definition
    """

    def test_rotation_matrix_x(self):
        """
        Test construction of rotation matrix by pi/2 radians around the
        x-axis
        """
        a = np.pi/2
        R = axis_angle_to_rotmat([1,0,0], a)
        Rref = np.array([[1,0,0],
                          [0,np.cos(a),-np.sin(a)],
                          [0,np.sin(a),np.cos(a)]])
        
        np.testing.assert_array_almost_equal_nulp(R, Rref)
        
    def test_rotation_matrix_y(self):
        """
        Test construction of rotation matrix by pi/2 radians around the
        y-axis
        """
        a = np.pi/2
        R = axis_angle_to_rotmat([0,1,0], a)
        Rref = np.array([[np.cos(a),0,np.sin(a)],
                          [0,1,0],
                          [-np.sin(a),0,np.cos(a)]])
        
        np.testing.assert_array_almost_equal_nulp(R, Rref)
        
    def test_rotation_matrix_z(self):
        """
        Test construction of rotation matrix by pi/2 radians around the
        z-axis
        """
        a = np.pi/2
        R = axis_angle_to_rotmat([0,0,1], a)
        Rref = np.array([[np.cos(a),-np.sin(a),0],
                          [np.sin(a),np.cos(a),0],
                          [0,0,1]])
        
        np.testing.assert_array_almost_equal_nulp(R, Rref)
        
    def test_euler_angles(self):
        """
        Test construction of Euler angles for pi radian rotation around the
        xyz-axis
        """
        axis = np.ones(3) / np.sqrt(3)
        angle = np.pi
        r = Rotation.from_rotvec(axis * angle)
        euler_angles = r.as_euler('zyz')
        
        alpha, beta, gamma = axis_angle_to_euler(axis, angle)
        
        np.testing.assert_almost_equal([alpha,beta,gamma], euler_angles)

if __name__ == '__main__':
    unittest.main()
