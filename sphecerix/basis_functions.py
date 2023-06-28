# -*- coding: utf-8 -*-

import numpy as np

class BasisFunction:
    
    def __init__(self, n, l, m):
        self.r = np.array([0,0,0])
        self.n = n
        self.l = l
        self.m = m
        self.atomid = None
        self.name = None
        
        self.name = self.__get_name()
        
    def __get_name(self):
        return str(self.n) + self.__get_type()
    
    def __get_type(self):
        results = [
            ['s'],
            ['py', 'pz', 'px'],
            ['dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2']
        ]
        
        return results[self.l][self.m + self.l]