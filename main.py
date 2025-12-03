import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#read points (x, y) from file
df = pd.read_csv('data.csv')

point_x = df.iloc[:, 0].to_numpy()
point_y = df.iloc[:, 1].to_numpy()
#---


def cholesky_decomp(A):
    # *assuming we have positive definite and square matrix A

    # L * Lt = A (get L)
    n = len(A)
    L = np.zeros_like(A, dtype=float)
    
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L

def forward_substitution(matrix, vector):
    # L * y = b (get y)
    n = len(matrix)
    y = np.zeros(n)
    
    for k in range(n):
        if k == 0:
            sum_of_terms = 0
        else:
            sum_of_terms = np.dot(matrix[k, :k], y[:k])
        
        y[k] = (vector[k] - sum_of_terms) / matrix[k, k]
    
    return y

def backward_substitution(matrix, vector):
    # U * x = y (get x)
    n = len(matrix)
    x = np.zeros(n)
    
    for k in range(1, n+1):
        if k == 1:
            sum_of_terms = 0
        else:
            sum_of_terms = np.dot(matrix[n-k, -(k-1):], x[-(k-1):])
        
        x[n-k] = (vector[n-k] - sum_of_terms) / matrix[n-k, n-k]
    
    return x



# A * x = B, find x vector (x = [a, b , c])
def quadraticReg(A, B):
    # To find vector x = [a, b, c] we need to:

    # 1. Cholesky decomposition: A = L * Lt
    L = cholesky_decomp(A)

    # 2. forward substitution on L (find y values) L * y = B
    y = forward_substitution(L, B)

    # 3. backward substitution on Lt (find x vector [a, b, c] values) Lt * x = y
    x = backward_substitution(L.T, y)

    #a, b, c = np.linalg.solve(A, B) #error check with automatic solve

    return x #[a, b, c]



#task: A * [a, b, c] = B, find quadratic coefficients a, b, c
#calculating matrix A - derivative of MSE for quadratic equation (calculated in class)
Sx = np.sum(point_x)
Sx2 = np.sum(point_x**2)
Sx3 = np.sum(point_x**3)
Sx4 = np.sum(point_x**4)

A = np.array([
        [Sx4, Sx3, Sx2],
        [Sx3, Sx2, Sx],
        [Sx2, Sx,  len(point_x)]
])
#---

#calculating vector B - derivative of MSE for quadratic equation (calculated in class)
Sy = np.sum(point_y)
Sxy = np.sum(point_x*point_y)
Sx2y = np.sum(point_x**2 * point_y)

B = np.array([Sx2y, Sxy, Sy])
#---



quad_coeffs = quadraticReg(A, B) #quad_coeffs[0] = a, quad_coeffs[1] = b, quad_coeffs[2] = c

# Smooth x values for the parabola
x_fit = np.linspace(min(point_x), max(point_x), 500)
y_fit = quad_coeffs[0] * x_fit**2 + quad_coeffs[1] * x_fit + quad_coeffs[2]

# Plot
plt.scatter(point_x, point_y, color='blue', label='Data points')   # original points
plt.plot(x_fit, y_fit, color='red', label='Fitted parabola')  # regression curve
plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Regression')
plt.legend()
plt.show()


