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
        
        # build expanded table
        self.expandedtable = np.zeros((self.nrgroups, self.order), dtype=np.int64)
        ctr = 0
        for i,g in enumerate(self.chartablelib['symmetry_groups']):
            idx = 0
            for j,c in enumerate(self.chartablelib['classes']):
                for k in range(c['multiplicity']):
                    self.expandedtable[i,idx] = g['characters'][j]
                    idx += 1
        
    def lot(self, traces):
        return np.round(self.expandedtable @ traces / self.order)
        