import unittest
import numpy as np
import sys
import os

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import functions
from sphecerix import tesseral_wigner_D, tesseral_wigner_D_mirror
from scipy.spatial.transform import Rotation as R

class TestAmmonia(unittest.TestCase):
    """
    Test the construction of tesseral transformation matrices for the ammonia
    molecule (C3v symmetry)
    """

    def test_rotation_c3(self):
        """
        The p-orbitals on N in ammonia belong to the A1 (z) and E (x,y) groups, 
        thus the character under C3 should be 0.
        """
        axes = [[0,0,1], [0,0,-1]]
        angle = np.radians(120)  # C3 rotation
        
        for axis in axes:
            axis /= np.linalg.norm(axis)
            Robj = R.from_rotvec(np.array(axis) * angle)
            T = tesseral_wigner_D(1, Robj)
            
            # test that determinant of the transformation matrix is 1.0
            # because the operation contains *no* reflection
            np.testing.assert_almost_equal(np.linalg.det(T), 1.0)
            
            # test the the matrix multiplied with its complex conjugate
            # equals the identity matrix
            np.testing.assert_almost_equal(T @ T.transpose(), np.identity(3))
            
            # test that trace equals the character under C3
            np.testing.assert_almost_equal(np.trace(T), 0)
            
            # assert that z-component gives a 1
            np.testing.assert_almost_equal(T[1,1], 1)
            
            # assert that x,y-components give -1
            np.testing.assert_almost_equal(T[0,0] + T[2,2], -1)
            
    def test_mirror_sigma_v(self):
        """
        The p-orbitals on N in ammonia belong to the A1 (z) and E (x,y) groups, 
        thus the character under sigma_v should be +1.
        """
        # construct the normal vectors for the sigma_v mirror planes. Given the
        # orientation of the molecule, one normal vector of a mirror plane lies
        # alongside the +x axis. The other normal vectors can be found using
        # 120 degree rotations of that vector
        normals = []
        for i in range(0,3):
            normals.append([np.cos(2.0 / 3.0 * i * np.pi), 
                            np.sin(2.0 / 3.0 * i * np.pi), 
                            0.0])
        
        for normal in normals:
            normal /= np.linalg.norm(normal)
            M = tesseral_wigner_D_mirror(1, normal)
            
            # test that determinant of the transformation matrix is -1.0
            # because the operation contains a reflection
            np.testing.assert_almost_equal(np.linalg.det(M), -1.0)
            
            # test the the matrix multiplied with its complex conjugate
            # equals the identity matrix
            np.testing.assert_almost_equal(M @ M.transpose(), np.identity(3))
                
            # test that trace equals the character under sigma_d
            np.testing.assert_almost_equal(np.trace(M), 1)
            
            # assert that z yields a 1 under sigma_d
            np.testing.assert_almost_equal(M[1,1], 1)
            
            # assert that x,y-components give 0 under sigma_d
            np.testing.assert_almost_equal(M[0,0] + M[2,2], 0)

    def build_coordinates_ammonia(self):
        """
        Construct fictitious ammonia coordinates
        """
        coords = [
            [ 0.00000000,    -0.00000000,    -0.06931370],
            [ 0.00000000,     0.94311105,     0.32106944],
            [-0.81675813,    -0.47155553,     0.32106944],
            [ 0.81675813,    -0.47155553,     0.32106944]
        ]
        
        return coords

if __name__ == '__main__':
    unittest.main()
