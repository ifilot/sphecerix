# -*- coding: utf-8 -*-

import json
import os
import numpy as np

class CharacterTable:
    """
    Class to store character table data in
    """
    def __init__(self, name):
        filename = os.path.join(os.path.dirname(__file__), 'charactertables', name + '.json')
        f = open(filename, 'r')
        self.chartablelib = json.load(f)
        self.order = np.sum([int(c['multiplicity']) for c in self.chartablelib['classes']])
        self.nrclasses = len(self.chartablelib['classes'])
        self.nrgroups = len(self.chartablelib['symmetry_groups'])
        
        # build condensed and expanded table
        self.table = np.zeros((self.nrgroups, self.nrgroups), dtype=np.float64)
        self.expandedtable = np.zeros((self.nrgroups, self.order), dtype=np.float64)
        for i,g in enumerate(self.chartablelib['symmetry_groups']): # loop over irreps
            idx = 0 # loop over operations
            for j,c in enumerate(self.chartablelib['classes']):
                self.table[i,j] = g['characters'][j]
                for k in range(c['multiplicity']):
                    self.expandedtable[i,idx] = g['characters'][j]
                    idx += 1
        
        # check that the character table is correct
        try:
            np.testing.assert_almost_equal(self.expandedtable @ self.expandedtable.transpose() / self.order, 
                                           np.identity(self.nrgroups))
        except AssertionError as e:
            print('Invalid character table encountered in: %s' % filename)
            raise e
        
    def lot(self, traces):
        return np.round(self.expandedtable @ traces / self.order)
    
    def get_label_irrep(self, irrep_idx):
        return self.chartablelib['symmetry_groups'][irrep_idx]['symbol']
    
    def get_character(self, irrep_idx, op_idx):
        return self.expandedtable[irrep_idx, op_idx]