import numpy as np
import matplotlib.pyplot as plt

"""
ASSIGNMENT1 using numpy
Write a program in Python that estimates the coefficients of the simple linear
regression equation, prints them, and plots the actual and estimated values (regression
line). Define the following functions: meanFun, varianceFun,
covarianceFun, coefficientsFun
USING numpy
"""

"""
[numpy Explanation]

np.cov = Estimate a covariance matrix, given data and weights.
np.var = Compute the variance along the specified axis.
np.mean = Compute the arithmetic mean along the specified axis.
np.sum = Sum of array elements over a given axis.
ddof = delta degrees of freedom, it is used for bias adjustment
ddof=1 will return the unbiased estimate.
ddof=0 will return the simple average.
"""


def coefficients_fun(x, y):
    # Returns the coefficients of the regression line between the two input arrays x and y.
    b1 = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    b0 = np.mean(y) - b1 * np.mean(x)
    return (b0, b1)


def rSquared(y, prediction_y):
    # Returns the coefficient of determination (r²) between the two input lists x and y.
    #
    # SST = Σ(yᵢ - ȳ)² = total sum of squares
    # SSR = Σ(ŷᵢ - ȳ)² = sum of squares due to regression
    # r² = SSR / SST, so r2 (which is r²) is the coefficient of determination

    sst = np.sum((y - np.mean(y)) ** 2)
    ssr = np.sum((prediction_y - np.mean(y)) ** 2)
    r2 = (ssr / sst)

    return r2


sample_data = np.array([[1, 14], [3, 24], [2, 18], [1, 17], [3, 27]])

# arrays takes elements from sampleData
array_X = sample_data[:, 0]
array_Y = sample_data[:, 1]

coefficient = coefficients_fun(array_X, array_Y)
prediction_Y = coefficient[0] + coefficient[1] * array_X

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
