# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter

class ProjectionOperator:
    
    def __init__(self, ct, so):
        #
        # ToDo: assert that the objects have the right type
        #
        
        self.ct = ct # character table
        self.so = so # symmetry operations
        self.groups = None
        self.irreps = None
        
    def collect(self):
        """
        Construct groups of basis functions that can transform among each
        other and figure out which irreps these span
        """
        # create one logical matrix that shows the interaction
        groupmatrix = np.zeros_like(self.so.operation_matrices[0], dtype=np.bool8)
        for m in self.so.operation_matrices:
            bm = np.abs(m) > 1e-9
            groupmatrix = np.logical_or(groupmatrix, bm)
        
        #print(groupmatrix)
        
        # collect the groups
        self.groups = []
        for i,row in enumerate(groupmatrix):
            if np.sum(row) > 1:
                idx = np.where(row)[0]
                self.groups.append(np.array(idx, dtype=np.int64))
            else:
                self.groups.append(np.array([i], dtype=np.int64))
        
        # create unique list
        self.groups = np.array(self.groups, dtype=object)
        res = Counter(map(tuple, self.groups))
        self.groups = [r for r in res]
        
        # determine irreps per lips
        self.irreps = []
        for g in self.groups:
            chars = [np.sum(np.take(np.diagonal(m),g)) for m in self.so.operation_matrices]
            self.irreps.append(self.ct.lot(chars))
        
    def build_mos(self):
        # check if groups have been collected
        if self.groups is None:
            self.collect()

        # build molecular orbitals
        self.mos = np.zeros((len(self.so.mol.basis), len(self.so.mol.basis)),
                            dtype=np.float64)
        mo_idx = 0
        for g,irreplist in zip(self.groups,self.irreps):
            for j,irrep in enumerate(irreplist): # loop over irreps
            
                # early exit if there are no irreps of this type
                if irrep == 0:
                    continue
            
                irrep_e = self.ct.get_character(j,0) # get dimensionality of irrep
                
                irrepres = np.zeros((int(irrep * irrep_e), len(self.so.mol.basis)))
                for k in range(int(irrep * irrep_e)): # loop over times the irrep is present
                    #label = self.ct.get_label_irrep(j)
                    res = self.apply_projection_operator(g[k], j)
                    irrepres[k,:] = res
                
                # perform orthogonalization if the irrep space is larger than 1
                if len(irrepres) > 1:
                    S = irrepres @ irrepres.transpose()
                    e,v = np.linalg.eigh(S)
                    X = v @ np.diag(1.0 / np.sqrt(e)) @ v.transpose()
                    irrepres = X @ irrepres
                    
                    for mo in irrepres:
                        self.mos[mo_idx] = mo
                        mo_idx += 1
                else:
                    self.mos[mo_idx] = irrepres[0,:]
                    mo_idx += 1
                     
        return self.mos
        
                    
    def apply_projection_operator(self, bf_idx, irrep_idx):
        """
        Apply the projection operator to a basis function
        """
        res = np.zeros(len(self.so.mol.basis))
        for i,m in enumerate(self.so.operation_matrices):
            res += self.ct.get_character(irrep_idx, i) * m[bf_idx,:]

        # return result and normalize it
        return np.array(res, dtype=np.float64) / np.linalg.norm(res)