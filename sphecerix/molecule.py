# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy

class Molecule:
    """
    Molecule class
    """
    def __init__(self, _name='unknown'):
        self.atoms = []
        self.charges = []
        self.name = _name
        self.basis = None

    def from_file(self, path, molname=None):
        """
        Build molecule from file and return it
        """
        self.name = molname

        with open(path, 'r') as f:
            lines = f.readlines()
            
            nratoms = int(lines[0].strip())

            for line in lines[2:2+nratoms]:
                pieces = line.split()
                self.add_atom(pieces[0], float(pieces[1]), float(pieces[2]), float(pieces[3]), unit='angstrom')

    def __str__(self):
        res = "Molecule: %s\n" % self.name
        for atom in self.atoms:
            res += " %s (%f,%f,%f)\n" % (atom[0], atom[1][0], atom[1][1], atom[1][2])

        return res

    def add_atom(self, atom, x, y, z, unit='bohr'):
        ang2bohr = 1.8897259886

        x = float(x)
        y = float(y)
        z = float(z)

        if unit == "bohr":
            self.atoms.append([atom, np.array([x, y, z])])
        elif unit == "angstrom":
            self.atoms.append([atom, np.array([x*ang2bohr, y*ang2bohr, z*ang2bohr])])
        else:
            raise RuntimeError("Invalid unit encountered: %s. Accepted units are 'bohr' and 'angstrom'." % unit)

        self.charges.append(0)
        
    def build_basis(self, molset):
        self.basis = []
        
        for i,atom in enumerate(self.atoms):
            if atom[0] in molset.keys():
                for bf in molset[atom[0]]:
                    abf = deepcopy(bf)
                    abf.name = atom[0] + bf.name
                    abf.r = np.array([atom[1][i] for i in range(3)])
                    abf.atomid = i
                    self.basis.append(abf)
            