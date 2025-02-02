import unittest
import numpy as np
import sys
import os

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import functions
from sphecerix import CharacterTable

class TestCharacterTable(unittest.TestCase):
    """
    Test loading character tables from json files
    """

    def test_character_table_c3v(self):
        ct = CharacterTable('c3v')
        
        self.assertEqual(ct.order, 6)
        
        np.testing.assert_almost_equal(ct.lot(np.ones(6)), [1,0,0])
        np.testing.assert_almost_equal(ct.lot([1,1,1,-1,-1,-1]), [0,1,0])
        np.testing.assert_almost_equal(ct.lot([4,-2,-2,0,0,0]), [0,0,2])

if __name__ == '__main__':
    unittest.main()
