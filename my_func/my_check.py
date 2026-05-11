import numpy as np

def is_hermitian(matrix):
    """Return True when a matrix equals its conjugate transpose."""
    is_hermitian = np.allclose(matrix, np.conj(matrix.T), atol=1e-10)

    if is_hermitian:
        return True
    else:
        return False

def is_unitary(matrix):
    """Return True when U dagger U is the identity within tolerance."""

    identity_matrix = np.eye(matrix.shape[0])

    return np.allclose(np.dot(np.conj(matrix.T), matrix), identity_matrix,atol=1e-13)
