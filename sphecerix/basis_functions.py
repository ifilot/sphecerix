# -*- coding: utf-8 -*-

import numpy as np

class BasisFunction:
    
    def __init__(self, n, l, m):
        self.r = np.array([0,0,0])
        self.n = n
        self.l = l
        self.m = m
        self.atomid = None