"""PROBLEM: Linear Regression Using Normal Equation (easy)
Write a Python function that performs linear regression using the normal equation. 
The function should take a matrix X (features) and a vector y (target) as input, and return the coefficients of the linear regression model. 
Round your answer to four decimal places, -0.0 is a valid result for rounding a very small number.
Example:
    input: X = [[1, 1], [1, 2], [1, 3]], y = [1, 2, 3]
    output: [0.0, 1.0]
    reasoning: The linear model is y = 0.0 + 1.0*x, perfectly fitting the input data"""

import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    """y = X θ => X.T y = (X.T X) θ => θ = (X.T X)^{-1} X.T y"""
    X = np.array(X)
    y = np.array(y)

    A = np.linalg.inv(np.dot(X.T, X))
    B = np.dot(A, X.T)
    theta = np.dot(B, y)
    return np.round(theta, 2)

if __name__=="__main__":
    X = [[1, 1], [1, 2], [1, 3]]
    y = [1, 2, 3]

    print(linear_regression_normal_equation(X, y))
    