# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter
import random
import networkx as nx

class ProjectionOperator:
    
    def __init__(self, ct, so, tolerance=1e-4):
        #
        # ToDo: assert that the objects have the right type
        #
        
        self.ct = ct # character table
        self.so = so # symmetry operations
        self.groups = None
        self.irreps = None
        self.block_sizes = None
        self.tolerance = tolerance
        
    def collect(self):
        """
        Construct groups of basis functions that can transform among each
        other and figure out which irreps these span
        """
        # Create one logical matrix that shows which basis functions are allowed
        # to mix with which other basis functions for any of the symmetry operations
        # that are part of the group.
        groupmatrix = np.zeros_like(self.so.operation_matrices[0], dtype=np.bool8)
        for m in self.so.operation_matrices:
            bm = np.abs(m) > self.tolerance
            groupmatrix = np.logical_or(groupmatrix, bm)
        
        # Build a graph for all basis functions that can be transformed into
        # each other
        G = nx.Graph()
        for i,row in enumerate(groupmatrix):
            G.add_node(i)
            if np.sum(row) > 1:
                idx = np.where(row)[0]
                for j in idx:
                    G.add_edge(i,j)
            
        self.groups = [tuple(c) for c in nx.connected_components(G)]
        print(self.groups)
        
    def build_mos(self, verbose=False):
        # check if groups have been collected
        if self.groups is None:
            self.collect()
        self.build_irreps()

        # build molecular orbitals
        self.mos = np.zeros((len(self.so.mol.basis), len(self.so.mol.basis)),
                            dtype=np.float64)
        mo_idx = 0
        for g,irreplist in zip(self.groups,self.irreps): # loop over the groups
        
            bf_idx = 0
        
            if verbose:
                print('Group: ', g)
                print('       ', [self.so.mol.basis[i].name for i in g])
                print('Irreps:')
        
            for j,irrep in enumerate(irreplist): # loop over irreps
            
                # early exit if there are no irreps of this type
                if irrep == 0:
                    continue
                
                if verbose:
                    label = self.ct.get_label_irrep(j)
                    print('  - %s: %i' % (label, irrep))
                    
                irrep_e = self.ct.get_character(j,0) # get dimensionality of irrep
                irrepres = np.zeros((int(irrep * irrep_e), len(self.so.mol.basis)))
                zero_modes = True

                while zero_modes: # keep on looping until no zero eigenvalues are found
                    bf_indices = random.sample(g, int(irrep * irrep_e))
                    for k in range(int(irrep * irrep_e)): # loop over times the irrep is present
                        #label = self.ct.get_label_irrep(j)
                        res = self.apply_projection_operator(bf_indices[k], j)
                        irrepres[k,:] = res
                        bf_idx += 1
                    
                        S = irrepres @ irrepres.transpose()
                        e,v = np.linalg.eigh(S)
                    
                        #print('Trying: ', e)
                        zero_modes = np.any(np.abs(e) < 1e-2)
                
                # perform orthogonalization if the irrep space is larger than 1
                if len(irrepres) > 1:
                    S = irrepres @ irrepres.transpose()
                    e,v = np.linalg.eigh(S)
                    #print(e)
                    
                    # for symmetric orthogonalization:
                    #    X = v @ np.diag(1.0 / np.sqrt(e)) @ v.transpose()
                    #    irrepres = X @ irrepres
                    # for canonical orthogonalization:
                    #    X = v @ np.diag(1.0 / np.sqrt(e))
                    #    irrepres = X.transpose() @ irrepres
                    
                    X = v @ np.diag(1.0 / np.sqrt(e))
                    irrepres = X.transpose() @ irrepres
                    
                    for mo in irrepres:
                        self.mos[mo_idx] = mo
                        mo_idx += 1
                else:
                    self.mos[mo_idx] = irrepres[0,:]
                    mo_idx += 1
                     
        return self.mos
        
    def build_irreps(self):
        # Determine irreps per unique group.
        self.irreps = []
        self.irreplabels = []
        self.block_sizes = []
        self.blocks = []
        for g in self.groups:
            chars = [np.sum(np.take(np.diagonal(m),g)) for m in self.so.operation_matrices]
            self.irreps.append(self.ct.lot(chars))
            
            for j,irrepdim in enumerate(self.irreps[-1]):
                if irrepdim > 0:
                    self.block_sizes.append(int(irrepdim * self.ct.chartablelib['symmetry_groups'][j]['characters'][0]))
                    self.blocks.append((self.block_sizes[-1], irrepdim, self.ct.get_label_irrep(j)))
                    
    def apply_projection_operator(self, bf_idx, irrep_idx):
        """
        Apply the projection operator to a basis function
        """
        res = np.zeros(len(self.so.mol.basis))
        for i,m in enumerate(self.so.operation_matrices):
            res += self.ct.get_character(irrep_idx, i) * m[bf_idx,:]

        # return result and normalize it
        return np.array(res, dtype=np.float64) / np.linalg.norm(res)
    
    def get_block_sizes(self):
        """
        Get the block sizes of the block-diagonal matrix after the matrix
        transformation in the MO basis
        """
        return self.block_sizes
    
    def get_blocks(self):
        return self.blocks
    
    def check_transformation(self):
        return
        newmats = [self.mos @ m @ self.mos.transpose() for m in self.so.operation_matrices]
        
        idx = 0
        for c in range(self.ct.nr_classes): # loop over classes
            for m in range(self.ct.classes[c]['multiplicity']):
                mat = newmats[idx]
                
                # check blocks for mat
                idx2 = 0
                
                idx += 1