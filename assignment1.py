import matplotlib.pyplot as plt

"""
ASSIGNMENT1
Write a program in Python that estimates the coefficients of the simple linear
regression equation, prints them, and plots the actual and estimated values (regression
line). Define the following functions: meanFun, varianceFun,
covarianceFun, coefficientsFun
"""


def meanFun(x):
    # Returns the mean of the input list x
    return sum(x) / len(x)


def varianceFun(x):
    # Returns the variance of the input list x.
    deviationSqrt = 0
    size = len(x)
    meanX = meanFun(x)
    for elementX in x:
        deviationSqrt += (elementX - meanX) ** 2
    variance = deviationSqrt / (size - 1)

    return variance


def covarianceFun(x, y):
    # Returns the covariance between the two input lists x and y.
    sumOfMultiplication = 0
    size = len(x)
    meanX = meanFun(x)
    meanY = meanFun(y)
    for i in range(0, size):
        sumOfMultiplication += (x[i] - meanX) * (y[i] - meanY)
    coVariance = sumOfMultiplication / (size - 1)

    return coVariance


def coefficientsFun(x, y):
    # Returns the coefficients of the regression line between the two input lists x and y.
    b1 = covarianceFun(x, y) / varianceFun(x)
    b0 = meanFun(y) - b1 * meanFun(x)

    return (b0, b1)


def rSquared(y, pred_Y):
    # Returns the coefficient of determination (r²) between the two input lists x and y.
    #
    # SST = Σ(yᵢ - ȳ)² = total sum of squares
    # SSR = Σ(ŷᵢ - ȳ)² = sum of squares due to regression
    # r2 = SSR / SST, so r2 (which is r²) is the coefficient of determination

    SST = 0
    SSR = 0

    for i in range(len(y)):
        SST += (y[i] - meanFun(y)) ** 2
    for i in range(len(y)):
        SSR += (pred_Y[i] - meanFun(y)) ** 2
    r2 = (SSR / SST)

    return r2


sampleData = [[1, 14], [3, 24], [2, 18], [1, 17], [3, 27]]

prediction_Y = []
array_X = []
array_Y = []

# arrays takes elements from sampleData
for element in sampleData:
    array_X.append(element[0])
    array_Y.append(element[1])

# coefficients of the regression line
coefficient = coefficientsFun(array_X, array_Y)
for elementX in array_X:
    prediction_Y.append(coefficient[0] + coefficient[1] * elementX)

# r2 is the coefficient of determination
r2 = rSquared(array_Y, prediction_Y)

# This plots the result and prints
plt.scatter(array_X, array_Y, s=40, color="g", marker='o')  # plots the sample data points in green dots
plt.scatter(array_X, prediction_Y, s=40, color="r", marker='o')  # plots the predicted values in red dots
plt.plot(array_X, prediction_Y, color="b")  # plots the regression line in blue
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

equation = "y = %.4f + %.4f * x" % (coefficient[0], coefficient[1])
print("b0 = %.4f , b1 = %.4f" % (coefficient[0], coefficient[1]))
print(equation)
print("r^2 = %.2f" % r2)
plt.text(2, 25, equation, fontsize=12)  # Writes y = 10.0000 + 5.0000 * x on the window
plt.text(2, 24, "r^2 = %.2f" % r2, fontsize=12)  # Writes r^2 = 0.88 on the window

plt.show()
# Please close window instead of stopping the process from here
