import math
import random

def matrixTranspose(matrix):
    transpoesd = []
    for i in range(0, len(matrix[0])):
        row = []
        for j in range(0, len(matrix)):
            row.append(matrix[j][i])
        transpoesd.append(row)
    return transpoesd


def matrixMinor(matrix, rowNum, columnNum):
    minor = []
    size = len(matrix)
    for i in range(0, size):
        row = []
        for j in range(0, size):
            if (j != columnNum and i != rowNum):
                row.append(matrix[i][j])
        if (row != []):
            minor.append(row)
    return minor


def matrixDeterminant(matrix):
    determinant = 0
    if (len(matrix) == 2):
        return (matrix[0][0] * matrix[1][1]) - (matrix[1][0] * matrix[0][1])
    for i in range(0, len(matrix)):
        determinant += matrixDeterminant(matrixMinor(matrix, 0, i)) * matrix[0][i] * ((-1) ** i)
    return determinant


def matrixInverse(matrix):
    adjointMatrix = []
    for i in range(0, len(matrix)):
        row = []
        for j in range(0, len(matrix)):
            row.append(matrixDeterminant(matrixMinor(matrix, i, j)) * ((-1) ** (i + j)) / matrixDeterminant(matrix))
        adjointMatrix.append(row)
    adjointMatrix = matrixTranspose(adjointMatrix)
    return adjointMatrix


def multi_matrices(arr1, arr2):
    arr_result = []
    k = 0
    if (len(arr1[0]) != len(arr2)):
        raise ValueError("The columns of first arr doesn't euqals to the rows of second arr")

    for i in range(len(arr1)):
        column_result = []
        for j in range(len(arr2[0])):
            sum = 0
            for k in range(len(arr1[0])):
                sum += arr1[i][k] * arr2[k][j]
            column_result.append(sum)
        arr_result.append(column_result)

    return arr_result


def fraction(numerator, denominator):
    gcd = math.gcd(numerator, denominator)
    return (str(numerator / gcd) + "/" + str(denominator / gcd))


def caculateX(matrixA, matrixB):
    if (len(matrixB[0]) != 1 and len(matrixB) != 1):
        raise ValueError("The columns of B or the row of B are both not 1")
    if (len(matrixB[0]) == 1):
        matrixX = multi_matrices(matrixInverse(matrixA), matrixB)
    else:
        matrixX = multi_matrices(matrixB, matrixInverse(matrixA))
    return matrixX



# Question a)

# Matrix from example
exampleA = [[4, 1, -5],
            [-2, 3, 1],
            [3, -1, 4]]

exampleB = [[8], [12], [5]]

# The functions
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
beta = [1, 2, 5]
X = []
noise = []
Y = []
for i in range(0, n):
    X.append([1, random.gauss(0, 1), random.gauss(0, 1)])
    noise.append(random.gauss(0, 1))
    Y.append([beta[0] + beta[1] * X[i][1] + beta[2] * X[i][2] + noise[i]])
print(X)
print(">> Random data generated with n =", n, ", k =", k, ", beta =", beta, "\n\n")

# Question d)
print("Question d: Estimation of the regression coefficients 𝒃 \n")
transposedX = matrixTranspose(X)
bValues = multi_matrices(multi_matrices(matrixInverse(multi_matrices(transposedX, X)), transposedX), Y)
print(">>", bValues)
