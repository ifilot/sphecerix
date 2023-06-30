import numpy as np
from scipy.spatial.transform import Rotation as R
from . import tesseral_wigner_D, tesseral_wigner_D_mirror, tesseral_wigner_D_improper

class SymmetryOperations:
    """
    Class containing all symmetry operations as applied to molecule object
    """ 
    def __init__(self, mol):
        self.mol = mol
        self.operations = []
        
        # build atoms matrix
        self.positions = np.zeros((len(self.mol.atoms), 3))
        for i,atom in enumerate(self.mol.atoms):
            self.positions[i,:] = atom[1]
        
    def add(self, name, label = None, vec = None, angle = None):
        # ensure vector is of float type
        if vec is not None:
            vec = np.array(vec, dtype=np.float64)
        
        if name == 'identity':
            self.operations.append(Identity())
        elif name == 'rotation':
            self.operations.append(Rotation(label, vec, angle))
        elif name == 'mirror':
            self.operations.append(Mirror(label, vec))
        elif name == 'improper':
            self.operations.append(ImproperRotation(label, vec, angle))
        elif name == 'inversion':
            self.operations.append(Inversion())
        else:
            raise Exception('Unknown operation: %s' % name)
    
    def run(self):
        N = len(self.mol.atoms)   # number of atoms
        nbf = len(self.mol.basis) # number of basis functions
        self.atomic_transformations = np.zeros((len(self.operations),N), dtype=np.uint64)
        self.basis_function_transformations = np.zeros((len(self.mol.basis), N, N))
        
        # assert atomic operations
        for k,operation in enumerate(self.operations):
            tpos = self.positions @ operation.get_matrix()
            
            for i,p in enumerate(self.positions):
                for j,t in enumerate(tpos):
                    r = t-p
                    if np.sum(r**2) < 1e-5:
                        self.atomic_transformations[k,i] = j
        
        # cache on which atoms basis functions are located
        bfindices = -np.ones((N, 5, 3, 7), dtype=np.int64) # encode all possibilities up to 5f-orbitals
        for i,bf in enumerate(self.mol.basis):
            bfindices[bf.atomid, bf.n, bf.l, bf.m+bf.l] = i
        
        # assert basis function operations
        self.operation_matrices = np.zeros((len(self.operations), nbf, nbf))
        for k,operation in enumerate(self.operations):
            for i,bf in enumerate(self.mol.basis):
                # establish on which atom id the basis function lands
                atomid = self.atomic_transformations[k,bf.atomid]
                
                # establish tesseral transformation
                mvec = np.zeros(2*bf.l+1)
                mvec[bf.m + bf.l] = 1
                mres = operation.get_wigner_matrix(bf.l).dot(mvec)
                
                for m,v in enumerate(mres):
                    idx = bfindices[atomid, bf.n, bf.l, m]
                    if idx == -1:
                        raise Exception('Cannot perform this operation')
                    self.operation_matrices[k,i,idx] = v

class Operation:
    """
    Base operation class
    """
    def __init__(self, name):
        self.name = name
        
    def set_atomic_id(self, idx):
        self.atomid = idx

class Identity(Operation):
    """
    Identity operation "E"
    """
    def __init__(self):
        super().__init__('E')

    def get_matrix(self):
        return np.identity(3)
    
    def get_wigner_matrix(self, l):
        return np.identity(2*l+1)
    
class Inversion(Operation):
    """
    Identity operation "i"
    """
    def __init__(self):
        super().__init__('i')

    def get_matrix(self):
        return -np.identity(3)
    
    def get_wigner_matrix(self, l):
        return np.identity(2*l+1) * (-1)**l

class Rotation(Operation):
    """
    Rotation operation "C"
    """
    def __init__(self, label, axis, angle):
        super().__init__('C' + label)
        self.axis = axis / np.linalg.norm(axis)
        self.angle = angle
        self.robj = R.from_rotvec(self.axis * self.angle)
        
    def get_matrix(self):
        return self.robj.as_matrix()
    
    def get_wigner_matrix(self, l):
        return tesseral_wigner_D(l, self.robj)
        
class Mirror(Operation):
    """
    Rotation operation "σ"
    """
    def __init__(self, label, normal):
        super().__init__('σ' + label)
        self.normal = normal
        
    def get_matrix(self):
        return np.identity(3) - 2 * np.outer(self.normal, self.normal)
    
    def get_wigner_matrix(self, l):
        return tesseral_wigner_D_mirror(l, self.normal)
        
class ImproperRotation(Operation):
    """
    Rotation operation "S"
    """
    def __init__(self, label, axis, angle):
        super().__init__('S' + label)
        self.axis = axis / np.linalg.norm(axis)
        self.angle = angle
        self.robj = R.from_rotvec(self.axis * self.angle)
        
    def get_matrix(self):
        M = np.identity(3) - 2 * np.outer(self.axis, self.axis)
        R = self.robj.as_matrix()
        return M @ R
    
    def get_wigner_matrix(self, l):
        return tesseral_wigner_D_improper(l, self.robj)