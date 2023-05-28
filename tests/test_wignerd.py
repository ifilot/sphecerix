import unittest
import numpy as np
import sys
import os
import warnings

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import functions
from sphecerix import wigner_D, tesseral_wigner_D
from sphecerix import tesseral_transformation,permutation_sh_car
from scipy.spatial.transform import Rotation as R

class TestWignerD(unittest.TestCase):    

    def test_wigner_D_z(self):
        """
        Test Wigner-D matrix for rotation around the z-axis by pi/2 radians
        """
        Robj = R.from_euler('zyz', [0.0, 0.0, np.pi/2])
        D = wigner_D(1, Robj)
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
        Robj = R.from_euler('zyz', [np.pi/2, np.pi/2, -np.pi/2])
        D = wigner_D(1, Robj)
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
        Robj = R.from_euler('zyz', [0, np.pi/2, 0])
        D = wigner_D(1, Robj)
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
        
        Robj = R.from_euler('zyz', [0, np.pi/2, 0])

        D = wigner_D(1, Robj)
        rm = P.transpose() @ np.real(T @ D @ T.conjugate().transpose()) @ P
        
        axis = [0,1,0]
        angle = np.pi/2
        Rmat = R.from_rotvec(np.array(axis) * angle).as_matrix()
        
        np.testing.assert_array_almost_equal(rm, Rmat)

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
        Robj = R.from_rotvec(np.array(axis) * angle)
        
        D = wigner_D(1, Robj)
        rm = P.transpose() @ np.real(T @ D @ T.conjugate().transpose()) @ P
        Rmat = Robj.as_matrix()
                
        np.testing.assert_array_almost_equal(rm, Rmat)
        
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
        Robj = R.from_rotvec(np.array(axis) * angle)
        
        D = wigner_D(1, Robj)
        rm = P.transpose() @ np.real(T @ D @ T.conjugate().transpose()) @ P
        Rmat = Robj.as_matrix()
                
        np.testing.assert_array_almost_equal(rm, Rmat)
        
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
        Robj = R.from_rotvec(np.array(axis) * angle)
        
        D = wigner_D(1, Robj)
        rm = P.transpose() @ np.real(T @ D @ T.conjugate().transpose()) @ P
        Rmat = Robj.as_matrix()
                
        np.testing.assert_array_almost_equal(rm, Rmat)

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
        
        Robj = R.from_rotvec(axis * angle)
        D = wigner_D(1, Robj)
        rm = P.transpose() @ np.real(T @ D @ T.conjugate().transpose()) @ P
        Rmat = Robj.as_matrix()
        
        np.testing.assert_array_almost_equal(rm, Rmat)
        
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
            Robj = R.from_rotvec(axis * angle)
            
            D = wigner_D(1, Robj)
            rm = P.transpose() @ np.real(T @ D @ T.conjugate().transpose()) @ P
            Rmat = Robj.as_matrix()
            
            np.testing.assert_array_almost_equal(rm, Rmat)
            
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
            Robj = R.from_rotvec(axis * angle)
            
            D = wigner_D(1, Robj)
            rm = P.transpose() @ np.real(T @ D @ T.conjugate().transpose()) @ P
            Rmat = Robj.as_matrix()
            
            np.testing.assert_array_almost_equal(rm, Rmat)
            
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
            Robj = R.from_rotvec(axis * angle)
            
            D = tesseral_wigner_D(1, Robj)
            rm = P.transpose() @ D @ P
            Rmat = Robj.as_matrix()
            
            np.testing.assert_array_almost_equal(rm, Rmat)
            
    def test_wigner_d_l2(self):  
        """
        Test construction of Wigner-D matrix for l=2
        """
        Robj = R.from_euler('zyz', [np.pi/2, np.pi/3, np.pi/4])
        D = wigner_D(2, Robj)
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.")
        unittest.main()
