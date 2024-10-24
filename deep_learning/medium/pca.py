# PROBLEM: Principal Component Analysis (PCA) Implementation (medium)
# Write a Python function that performs Principal Component Analysis (PCA) from scratch. 
# The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature. 
# The function should standardize the dataset, compute the covariance matrix, find the eigenvalues and eigenvectors, and return the principal components (the eigenvectors corresponding to the largest eigenvalues). 
# The function should also take an integer k as input, representing the number of principal components to return.

# Example:
#     input: data = np.array([[1, 2], [3, 4], [5, 6]]), k = 1
#     output:  [[0.7071], [0.7071]]
#     reasoning: After standardizing the data and computing the covariance matrix, the eigenvalues and eigenvectors are calculated. 
# The largest eigenvalue's corresponding eigenvector is returned as the principal component, rounded to four decimal places.

import numpy as np

def pca(data: np.ndarray, k: int) -> list[list[int|float]]:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
	
    covariance_matrix = np.cov(standardized_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    idx = eigenvalues.argsort()[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]

    principal_components = sorted_eigenvectors[:, :k].tolist()
    return principal_components

if __name__ == "__main__":
    print(pca(np.array([[1, 2], [3, 4], [5, 6]]), k=1))
    print(pca(np.array([[4,2,1],[5,6,7],[9,12,1],[4,6,7]]),2))
