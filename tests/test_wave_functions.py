# -*- coding: utf-8 -*-

import numpy as np
from pylebedev import PyLebedev
import unittest
import sys
import os

# add a reference to load the Sphecerix library
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sphecerix import wfcart

class TestAtomicWaveFunctions(unittest.TestCase):

    def test_ao_normalized(self):
        """
        Test that the atomic wave functions are normalized
        """
        # use Gauss Chebychev Lebedev quadrature to assess normalization
        weights, points = self.__construct_gcl_grid(128, 35)
        
        # evaluate AOs on grid
        for n in range(1,5):
            for l in range(0,n):
                for m in range(-l,l+1):
                    vals = wfcart(n,l,m,points[:,0], points[:,1], points[:,2])
                    integral = np.sum(weights * vals * vals)
                    np.testing.assert_almost_equal(integral, 1.0, decimal=4)

    def __construct_gcl_grid(self,radial_points, lebedev_order):
        """
        Perform Gauss-Chebychev-Lebedev quadrature on trial wave function
        """
        # create grid points
        N = radial_points   # number of grid points
        rm = 0.35
    
        # build the Gauss-Chebychev grid following the canonical recipe
        z = np.arange(1, N+1)
        x = np.cos(np.pi / (N+1) * z)
        r = rm * (1 + x) / (1 - x)
        wr = np.pi / (N+1) * np.sin(np.pi / (N+1) * z)**2 * 2.0 * rm \
            / (np.sqrt(1 - x**2) * (1 - x)**2)
    
        # get Lebedev points
        leblib = PyLebedev()
        p,wl = leblib.get_points_and_weights(lebedev_order)
    
        # construct full grid
        gridpts = np.outer(r, p).reshape((-1,3))
        gridw = np.outer(wr * r**2, wl).flatten() * 4.0 * np.pi
    
        return gridw, gridpts

if __name__ == '__main__':
    unittest.main()
