import unittest
import numpy as np
import sys
import os

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import functions
from sphecerix import Molecule, BasisFunction, SymmetryOperations,\
                      CharacterTable

class TestBenzeneSymmetryOperations(unittest.TestCase):
    """
    Test all the symmetry operations under ethylene (D2h)
    """

    def test_benzene(self):
        mol = Molecule()
        for i in range(6):
            x = 1.3868467444 * np.sin(2.0 * np.pi / 6. * i)
            y = -1.3868467444 * np.cos(2.0 * np.pi / 6. * i)
            mol.add_atom('C', x, y, 0.0, unit='angstrom')

        for i in range(6):
            x = 2.4694205285 * np.sin(2.0 * np.pi / 6. * i)
            y = -2.4694205285 * np.cos(2.0 * np.pi / 6. * i)
            mol.add_atom('H', x, y, 0.0, unit='angstrom')
        
        molset = {
            'C': [BasisFunction(2,0,0)],
            'H': [BasisFunction(1,0,0)]
        }
        mol.build_basis(molset)
        
        symops = SymmetryOperations(mol)
        
        # E
        symops.add('identity')
        
        # 2C6
        symops.add('rotation', '6+', np.array([0,0,1]), 2.0 * np.pi / 6)
        symops.add('rotation', '6-', np.array([0,0,1]), -2.0 * np.pi / 6)
        
        # 2C3
        symops.add('rotation', '3+', np.array([0,0,1]), 2.0 * np.pi / 3)
        symops.add('rotation', '3-', np.array([0,0,1]), -2.0 * np.pi / 3)
        
        # C2
        symops.add('rotation', '2', np.array([0,0,1]), np.pi)
        
        # 3C2'
        for i in range(0,3):
            symops.add('rotation', '2,%i' % i, np.array([np.sin(2.0 * np.pi * i/6.),
                                                          -np.cos(2.0 * np.pi * i/6.),
                                                          0.0]), np.pi)
        
        # 3C2''
        for i in range(0,3):
            symops.add('rotation', '2,%i' % i, np.array([np.sin(2.0 * np.pi * (i/6. + 1./12)),
                                                          -np.cos(2.0 * np.pi * (i/6. + 1./12)),
                                                          0.0]), np.pi)
        
        
        # inversion
        symops.add('inversion')
        
        # 2S3
        symops.add('improper', '3+', np.array([0,0,1]), 2.0 * np.pi / 3)
        symops.add('improper', '3-', np.array([0,0,1]), -2.0 * np.pi / 3)
        
        # 2S6
        symops.add('improper', '6+', np.array([0,0,1]), 2.0 * np.pi / 6)
        symops.add('improper', '6-', np.array([0,0,1]), -2.0 * np.pi / 6)
        
        # sigma_h
        symops.add('mirror', 'h(xy)', np.array([0,0,1]))
        
        # sigma_d
        for i in range(0,3):
            symops.add('mirror', 'd,%i' % i, np.array([np.cos(2.0 * np.pi * (i/6. + 1./12)),
                                                       np.sin(2.0 * np.pi * (i/6. + 1./12)),
                                                       0.0]))
        
        # sigma_v
        for i in range(0,3):
            symops.add('mirror', 'v,%i' % i, np.array([np.cos(2.0 * np.pi * i/6),
                                                       np.sin(2.0 * np.pi * i/6),
                                                       0.0]))

        symops.run()
        
        # identity operation
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[0])[0:6], 
                                       [0,1,2,3,4,5])
        
        # C6+
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[1])[0:6], 
                                       [1,2,3,4,5,0])
        
        # C6-
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[2])[0:6], 
                                       [5,0,1,2,3,4])
        
        # C3+
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[3])[0:6], 
                                       [2,3,4,5,0,1])
        
        # C3-
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[4])[0:6], 
                                       [4,5,0,1,2,3])
        
        # C2
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[5])[0:6], 
                                       [3,4,5,0,1,2])
        
        # C2',1
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[6])[0:6], 
                                        [0,5,4,3,2,1])
        
        # C2',2
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[7])[0:6], 
                                        [2,1,0,5,4,3])
        
        # C2',3
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[8])[0:6], 
                                        [4,3,2,1,0,5])
        
        # C2'',1
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[9])[0:6], 
                                        [1,0,5,4,3,2])
        
        # C2'',2
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[10])[0:6], 
                                        [3,2,1,0,5,4])
        
        # C2'',3
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[11])[0:6], 
                                        [5,4,3,2,1,0])
        
        # inversion operation
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[12])[0:6], 
                                       [3,4,5,0,1,2])
        
        # S3+
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[13])[0:6], 
                                       [2,3,4,5,0,1])
        
        # S3-
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[14])[0:6], 
                                       [4,5,0,1,2,3])
        
        # S6+
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[15])[0:6], 
                                       [1,2,3,4,5,0])
        
        # S6-
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[16])[0:6], 
                                       [5,0,1,2,3,4])
        
        # sigma_h
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[17])[0:6], 
                                       [0,1,2,3,4,5])
        
        # sigma_d,1
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[18])[0:6], 
                                        [1,0,5,4,3,2])
        
        # sigma_d,2
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[19])[0:6], 
                                        [3,2,1,0,5,4])
        
        # sigma_d,3
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[20])[0:6], 
                                        [5,4,3,2,1,0])
                
        # sigma_v,1
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[21])[0:6], 
                                        [0,5,4,3,2,1])
        
        # sigma_v,2
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[22])[0:6], 
                                        [2,1,0,5,4,3])
        
        # sigma_v,3
        np.testing.assert_almost_equal(self.find_positions(symops.operation_matrices[23])[0:6], 
                                        [4,3,2,1,0,5])

    def find_positions(self, mat):
        indices = np.zeros(len(mat))
        for i in range(len(mat)):
            idx = np.where(np.abs(mat[i]) > 0.999)
            indices[i] = idx[0]
        return indices

if __name__ == '__main__':
    unittest.main()
