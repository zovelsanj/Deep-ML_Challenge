# PROBLEM: Cross-Validation Data Split Implementation (medium)
# Write a Python function that performs k-fold cross-validation data splitting from scratch. The function should take a dataset (as a 2D NumPy array where each row represents a data sample and each column represents a feature) and an integer k representing the number of folds. The function should split the dataset into k parts, systematically use one part as the test set and the remaining as the training set, and return a list where each element is a tuple containing the training set and test set for each fold.
# Example
# Example:
#     input: data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), k = 5
#     output: [[[[3, 4], [5, 6], [7, 8], [9, 10]], [[1, 2]]],
#             [[[1, 2], [5, 6], [7, 8], [9, 10]], [[3, 4]]],
#             [[[1, 2], [3, 4], [7, 8], [9, 10]], [[5, 6]]], 
#             [[[1, 2], [3, 4], [5, 6], [9, 10]], [[7, 8]]], 
#             [[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10]]]]

import numpy as np

def cross_validation_split(data: np.ndarray, k: int, seed=42) -> list[tuple[np.ndarray, np.ndarray]]:
    np.random.seed(seed)
    np.random.shuffle(data)
    
    folds = np.array_split(data, k)
    
    cross_val_sets = []
    for i in range(k):
        test_set = folds[i]
        train_set = np.vstack([fold for j, fold in enumerate(folds) if j != i])
        cross_val_sets.append((train_set.tolist(), test_set.tolist()))
    
    return cross_val_sets


if __name__ == "__main__":
    print(cross_validation_split(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), k=5, seed=42))
    print(cross_validation_split(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), k=2, seed=42))
    print(cross_validation_split(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]), k=3, seed=42))
