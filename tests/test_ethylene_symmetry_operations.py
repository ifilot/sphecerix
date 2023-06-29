import unittest
import numpy as np
import sys
import os

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import functions
from sphecerix import Molecule, BasisFunction, SymmetryOperations

class TestEthyleneSymmetryOperations(unittest.TestCase):
    """
    Test all the symmetry operations under ethylene (D2h)
    """

    def test_ethylene(self):
        mol = Molecule()
        mol.add_atom('C', -0.6530176758,  0.0000000000 ,0.0000000000, unit='angstrom')
        mol.add_atom('C',  0.6530176758,  0.0000000000 ,0.0000000000, unit='angstrom')
        mol.add_atom('H', -1.2288875372, -0.9156191261 ,0.0000000000, unit='angstrom')
        mol.add_atom('H', -1.2288875372,  0.9156191261 ,0.0000000000, unit='angstrom')
        mol.add_atom('H',  1.2288875372,  0.9156191261 ,0.0000000000, unit='angstrom')
        mol.add_atom('H',  1.2288875372, -0.9156191261 ,0.0000000000, unit='angstrom')
        
        molset = {
            'C': [BasisFunction(1,0,0),
                  BasisFunction(2,0,0),
                  BasisFunction(2,1,1),
                  BasisFunction(2,1,-1),
                  BasisFunction(2,1,0)],
            'H': [BasisFunction(1,0,0)]
        }
        mol.build_basis(molset)
        
        symops = SymmetryOperations(mol)
        symops.add('identity')
        symops.add('rotation', '2(z)', np.array([0,0,1]), np.pi)
        symops.add('rotation', '2(y)', np.array([0,1,0]), np.pi)
        symops.add('rotation', '2(x)', np.array([1,0,0]), np.pi)
        symops.add('inversion')
        symops.add('mirror', 'v(xy)', np.array([0,0,1]))
        symops.add('mirror', 'v(xz)', np.array([0,1,0]))
        symops.add('mirror', 'v(yz)', np.array([1,0,0]))

        symops.run()
        
        result = np.load(os.path.join(os.path.dirname(__file__), 'results', 'ethylene.npy'))
        np.testing.assert_almost_equal(symops.operation_matrices, result)

if __name__ == '__main__':
    unittest.main()
