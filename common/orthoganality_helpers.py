import numpy as np

def hadamard_matrix(n):
    """Recursively generate a Walsh-Hadamard matrix of size n x n.
       n must be a power of 2."""
    if n == 1:
        return np.array([[1]])
    else:
        H = hadamard_matrix(n // 2)
        top = np.concatenate((H, H), axis=1)
        bottom = np.concatenate((H, -H), axis=1)
        return np.concatenate((top, bottom), axis=0)

def walsh_hadamard(order):
    """Generate Walsh-Hadamard spreading codes of a given order.
       Each row of the matrix is a valid code."""
    H = hadamard_matrix(order)
    codes = [H[i] for i in range(H.shape[0])]
    return codes