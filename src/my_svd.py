# A new model! 

import numpy as np

# Define a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Perform SVD
U, s, Vh = np.linalg.svd(A)

print("U matrix:\n", U)
print("\nSingular values (s):\n", s)
print("\nVh matrix (V transpose):\n", Vh)

# Reconstruct the original matrix (optional)
# Create a diagonal matrix from singular values
Sigma = np.zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[1], :A.shape[1]] = np.diag(s)

B = U @ Sigma @ Vh
print("\nReconstructed matrix B:\n", B)
