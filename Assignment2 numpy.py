import math
import numpy as np

# transposed function
def matrixTranspose(matrix):
    return np.transpose(matrix)

# minor function
def matrixMinor(matrix, rowNum, columnNum):
    submatrix = np.delete(np.delete(matrix, rowNum, axis=0), columnNum, axis=1)
    return submatrix


# determinant function
def matrixDeterminant(matrix):
    determinant = np.linalg.det(matrix)
    return determinant


# inverse function
def matrixInverse(matrix):
    inverse = np.linalg.inv(matrix)
    return inverse

# caculate X function
def caculateX(matrixA, matrixB):
    if matrixB.shape[1] != 1 and matrixB.shape[0] != 1:
        raise ValueError("The columns or rows of matrixB are not equal to 1")
    if matrixB.shape[1] == 1:
        matrixX = np.dot(matrixInverse(matrixA), matrixB)
    else:
        matrixX = np.dot(matrixB, matrixInverse(matrixA))
    return matrixX




# Question a)

# Matrix from example
exampleA = np.array([[4, 1, -5], [-2, 3, 1], [3, -1, 4]])

exampleB = np.array([[8], [12], [5]])

# The functions to the result of the minor, transposed, determinant and inverse
minorA = matrixMinor(exampleA, 0, 0)
transposedA = matrixTranspose(exampleA)
determinantA = matrixDeterminant(exampleA)
inversedA = matrixInverse(exampleA)


print("Question a: Define 4 functions, and find the inverse of matrix A\n")
print("Transposed (matrixTranspose): ")
for element in transposedA:
    print(element)

print("\nMinor test (matrixMinor):")
for element in minorA:
    print(element)

print("\nDeterminant (matrixDeterminant):", determinantA, "\n")
print("Inversed matrix A (matrixInverse):")
for element in inversedA:
    print(element)
print("\n")

# Question b)
print("Question b: Solve the system 𝑨∙𝑿=𝑩 given the column vector of constants 𝑩=[𝟖,𝟏𝟐,𝟓] \n")
print(">>", caculateX(exampleA, exampleB), "\n\n")

# Question c)
print("Question c:  \n")
n = 50 # Bigger size will increase the accuracy of the estimation
k = 2
beta = np.array([1, 2, 5])
X = np.random.randn(n, k)
X = np.hstack((np.ones((n, 1)), X))
print(X)
noise = np.random.randn(n)
Y = X.dot(beta) + noise
print(">> Random data generated with n =", n, ", k =", k, ", beta =", beta, "\n\n")

# Question d)
print("Question d: Estimation of the regression coefficients 𝒃 \n")
transposedX = matrixTranspose(X)
bValues = np.dot(np.dot(matrixInverse(np.dot(transposedX, X)), transposedX), Y)
print(">>", bValues)
