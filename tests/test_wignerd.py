import unittest
import numpy as np
import sys
import os

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import functions
from sphecerix import axis_angle_to_euler, axis_angle_to_rotmat
from sphecerix import wigner_D, wigner_d, wigner_D_axis_angle, \
                      tesseral_wigner_D_axis_angle
from sphecerix import tesseral_transformation,permutation_sh_car

class TestWignerD(unittest.TestCase):    

    def test_wigner_D_z(self):
        """
        Test Wigner-D matrix for rotation around the z-axis by pi/2 radians
        """
        D = wigner_D(1, 0.0, 0.0, np.pi/2)
        Dref = np.array([
            [-1j,0,0],
            [0,1,0],
            [0,0,1j]
        ], dtype=np.complex128)
        
        np.testing.assert_array_almost_equal(D, Dref)
    
    def test_wigner_D_x(self):
        """
        Test Wigner-D matrix for rotation around the x-axis by pi/2 radians
        """
        D = wigner_D(1, np.pi/2, np.pi/2, -np.pi/2)
        Dref = np.array([
            [0.5, 1j/np.sqrt(2),-0.5],
            [1j/np.sqrt(2),0,1j/np.sqrt(2)],
            [-0.5,1j/np.sqrt(2),0.5]
        ], dtype=np.complex128)
        
        np.testing.assert_array_almost_equal(D, Dref)
        
    def test_wigner_D_y(self):
        """
        Test Wigner-D matrix for rotation around the y-axis by pi/2 radians
        """
        D = wigner_D(1, 0, np.pi/2, 0)
        Dref = np.array([
            [0.5, 1/np.sqrt(2),0.5],
            [-1/np.sqrt(2),0,1/np.sqrt(2)],
            [0.5,-1/np.sqrt(2),0.5]
        ], dtype=np.complex128)
        
        np.testing.assert_array_almost_equal(D, Dref)
        
    def test_wigner_D_y_cart(self):
        """
        Test Wigner-D matrix for rotation around the y-axis by pi/2 radians,
        followed by tesseral transformation
        """
        # construct tesseral transformation matrix
        T = tesseral_transformation(1)
        
        # grab yzx -> xyz transformation matrix
        P = permutation_sh_car()
        
        D = wigner_D(1, 0, np.pi/2, 0)
        rm = P.transpose() @ np.real(T @ D @ T.conjugate().transpose()) @ P
        
        R = axis_angle_to_rotmat([0,1,0], np.pi/2)
        
        np.testing.assert_array_almost_equal(rm, R)

    def test_rotations_l1_x(self):
        """
        Test Wigner-D matrix for rotation around the x-axis by pi/2 radians,
        followed by tesseral transformation and cartesian permutation such
        that result can be checked with the rotation matrix in R3
        """
        # construct tesseral transformation matrix
        T = tesseral_transformation(1)
        
        # grab yzx -> xyz transformation matrix
        P = permutation_sh_car()
        
        axis = [1,0,0]
        angle = np.pi / 2
        
        alpha, beta, gamma = axis_angle_to_euler(axis, angle)
        D = wigner_D(1, alpha, beta, gamma)
        rm = P.transpose() @ np.real(T @ D @ T.conjugate().transpose()) @ P
        R = axis_angle_to_rotmat(axis, angle)
                
        np.testing.assert_array_almost_equal(rm, R)
        
    def test_rotations_l1_z(self):
        """
        Test Wigner-D matrix for rotation around the x-axis by pi/2 radians,
        followed by tesseral transformation and cartesian permutation such
        that result can be checked with the rotation matrix in R3
        """
        # construct tesseral transformation matrix
        T = tesseral_transformation(1)
        
        # grab yzx -> xyz transformation matrix
        P = permutation_sh_car()
        
        axis = [0,0,1]
        angle = np.pi / 2
        
        alpha, beta, gamma = axis_angle_to_euler(axis, angle)
        D = wigner_D(1, alpha, beta, gamma)
        rm = P.transpose() @ np.real(T @ D @ T.conjugate().transpose()) @ P
        R = axis_angle_to_rotmat(axis, angle)
        
        np.testing.assert_array_almost_equal(rm, R)
        
    def test_rotations_l1_xy(self):
        """
        Test Wigner-D matrix for rotation around the xy-axis by pi radians,
        followed by tesseral transformation and cartesian permutation such
        that result can be checked with the rotation matrix in R3
        """
        # construct tesseral transformation matrix
        T = tesseral_transformation(1)
        
        # grab yzx -> xyz transformation matrix
        P = permutation_sh_car()
        
        axis = [1,1,0]
        axis /= np.linalg.norm(axis)
        angle = np.pi
        
        alpha, beta, gamma = axis_angle_to_euler(axis, angle)
        D = wigner_D(1, alpha, beta, gamma)
        rm = P.transpose() @ np.real(T @ D @ T.conjugate().transpose()) @ P
        R = axis_angle_to_rotmat(axis, angle)
        
        np.testing.assert_array_almost_equal(rm, R)

    def test_rotations_l1_xyz_cart(self):
        """
        Test Wigner-D matrix for rotation around the xyz-axis by pi radians,
        followed by tesseral transformation and cartesian permutation such
        that result can be checked with the rotation matrix in R3
        """
        # construct tesseral transformation matrix
        T = tesseral_transformation(1)
        
        # grab yzx -> xyz transformation matrix
        P = permutation_sh_car()
        
        axis = np.ones(3) / np.sqrt(3)
        angle = np.pi
        
        alpha, beta, gamma = axis_angle_to_euler(axis, angle)
        D = wigner_D(1, alpha, beta, gamma)
        rm = P.transpose() @ np.real(T @ D @ T.conjugate().transpose()) @ P
        R = axis_angle_to_rotmat(axis, angle)
        
        np.testing.assert_array_almost_equal(rm, R)
        
    def test_rotations_l1_random(self):
        """
        Test Wigner-D matrix for rotation around the randomnly generated
        axis and randomnly generated angle. Because everything is carried out
        for l=1, we can readily compare against rotation matrices.
        """
        # construct tesseral transformation matrix
        T = tesseral_transformation(1)
        
        # grab yzx -> xyz transformation matrix
        P = permutation_sh_car()
        
        # construct dedicated generator
        rng = np.random.default_rng(seed=42)
        
        for i in range(0,100):
            axis = rng.random(3)
            axis /= np.linalg.norm(axis)
            angle = rng.random(1) * 2.0 * np.pi
            
            alpha, beta, gamma = axis_angle_to_euler(axis, angle)
            D = wigner_D(1, alpha, beta, gamma)
            rm = P.transpose() @ np.real(T @ D @ T.conjugate().transpose()) @ P
            R = axis_angle_to_rotmat(axis, angle)
            
            np.testing.assert_array_almost_equal(rm, R)
            
    def test_rotations_l1_random_axis_angle(self):
        """
        Test Wigner-D matrix for rotation around the randomnly generated
        axis and randomnly generated angle. Because everything is carried out
        for l=1, we can readily compare against rotation matrices.
        """
        # construct tesseral transformation matrix
        T = tesseral_transformation(1)
        
        # grab yzx -> xyz transformation matrix
        P = permutation_sh_car()
        
        # construct dedicated generator
        rng = np.random.default_rng(seed=42)
        
        for i in range(0,100):
            axis = rng.random(3)
            axis /= np.linalg.norm(axis)
            angle = rng.random(1) * 2.0 * np.pi
            
            D = wigner_D_axis_angle(1, axis, angle)
            rm = P.transpose() @ np.real(T @ D @ T.conjugate().transpose()) @ P
            R = axis_angle_to_rotmat(axis, angle)
            
            np.testing.assert_array_almost_equal(rm, R)
            
    def test_tesseral_wigner_d_l1(self):
        """
        Test tesseral Wigner-D matrix for rotation around the randomnly 
        generated axis and randomnly generated angle. Because everything is 
        carried out for l=1, we can readily compare against rotation matrices.
        """
        # grab yzx -> xyz transformation matrix
        P = permutation_sh_car()
        
        # construct dedicated generator
        rng = np.random.default_rng(seed=42)
        
        for i in range(0,100):
            axis = rng.random(3)
            axis /= np.linalg.norm(axis)
            angle = rng.random(1) * 2.0 * np.pi
            
            D = tesseral_wigner_D_axis_angle(1, axis, angle)
            rm = P.transpose() @ D @ P
            R = axis_angle_to_rotmat(axis, angle)
            
            np.testing.assert_array_almost_equal(rm, R)
            
    def test_wigner_d_l2(self):  
        """
        Test construction of Wigner-D matrix for l=2
        """
        D = wigner_D(2, np.pi/2, np.pi/3, np.pi/4)
        Dref = np.array([
            [
                 0.5625j,	
                -0.649519052838,
                -0.459279326772j,
                 0.216506350946,
                 0.0625j
            ],
            [
                 0.459279326772-0.459279326772j,
                 0.0,
                 0.375 - 0.375j,
                 0.353553390593 + 0.353553390593j,
                -0.153093108924 + 0.153093108924j
            ],
            [
                -0.459279326772,	
                 0.53033008589j,
                -0.125,
                 0.53033008589j,
                -0.459279326772
            ],
            [
                 0.153093108924 + 0.153093108924j,
                 0.353553390593 - 0.353553390593j,
                -0.375 - 0.375j,
                 0.0,
                -0.459279326772 - 0.459279326772j
            ],
            [
                -0.0625j,
                -0.216506350946,
                0.459279326772j,
                0.649519052838,
                -0.5625j
            ]
        ], dtype=np.complex128)
        
        np.testing.assert_array_almost_equal(D, Dref)

if __name__ == '__main__':
    unittest.main()
