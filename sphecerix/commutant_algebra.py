# -*- coding: utf-8 -*-

import numpy as np
import scipy

def compute_commutant_basis(matrices, tol=1e-12):
    """
    Compute a numerical basis for the commutant (centralizer) of a set of square
    matrices. That is, find all matrices X such that [X, A] = 0 for all A in
    `matrices`.

    This is achieved by solving a homogeneous linear system using the Kronecker
    formulation: vec([X, A]) = (I ⊗ A - A.T ⊗ I) vec(X) = 0.

    Parameters:
    -----------
    matrices : list of ndarray
        List of n x n matrices (must be square and of equal shape).
    tol : float
        Tolerance for detecting the numerical rank (used in null space
        computation).

    Returns:
    --------
    commutant_basis : list of ndarray
        List of n x n matrices forming a basis for the commutant.
    """
    n = matrices[0].shape[0]

    # Build the constraint matrix from the Kronecker commutator identities
    constraints = [
        np.kron(np.eye(n), A) - np.kron(A.T, np.eye(n))
        for A in matrices
    ]
    full_constraint = np.vstack(constraints)

    # Compute the null space (each column is a basis vector for vec(X))
    null_basis = scipy.linalg.null_space(full_constraint, rcond=tol)

    # Reshape each basis vector back into an n x n matrix
    commutant_basis = [v.reshape((n, n)) for v in null_basis.T]

    return commutant_basis

import numpy as np

def construct_similarity_matrix(commutant_basis):
    """
    Given a list of symmetric, commuting matrices (the commutant basis), compute
    a similarity matrix that jointly diagonalizes them.

    This is achieved by averaging the matrices (symmetrized) and computing the
    eigenvectors of the resulting matrix. The output is a matrix U such that U.T
    @ A @ U is diagonal for all A in the commutant (ideally, if they truly
    commute and are symmetric).

    Parameters:
    -----------
    commutant_basis : list of ndarray
        List of n x n symmetric, commuting matrices.

    Returns:
    --------
    U : ndarray
        n x n orthogonal matrix whose columns are the joint eigenvectors.
    """
    # Symmetric average to ensure numerical stability
    A = sum((B + B.T) / 2 for B in commutant_basis)

    # Eigen decomposition (orthogonal U)
    _, U = np.linalg.eigh(A)

    return U
