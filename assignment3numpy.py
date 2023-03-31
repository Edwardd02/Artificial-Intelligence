import numpy as np

"""
ASSIGNMENT3 using numpy
Write a program in Python that estimates the regression coefficients 𝒃 by applying
the gradient descent algorithm.
Define the function:
gradientDescentFunction(data,iterations,learningRate).
Generate some random data of the form:
y₁ = β0 + β₀ + β₁x₁₁ + β₂x₁₂ + ⋯ + βₖx₁ₖ + ε₁
y₂ = β0 + β₀ + β₁x₂₁ + β₂x₂₂ + ⋯ + βₖx₂ₖ + ε₂
yₙ = β0 + β₀ + β₁xₙ₁ + β₂xₙ₂ + ⋯ + βₖxₙₖ + εₙ
where the dimensions of the matrix are: 𝒏 = 𝟏𝟎, 𝒌 = 𝟐, and 𝜷 = [𝟏, 𝟐, 𝟓].
Then make an estimation of the regression coefficients 𝒃
by applying the gradient descent algorithm.
Define the function:
gradientDescentFunction(data,iterations,learningRate)
USING numpy.
"""

def gradientDescentFunction(data, iterations, learningRate):
    x_Values = data[:, :-1]
    y_Values = data[:, -1:]
    n = len(y_Values)

    beta = np.zeros((x_Values.shape[1], 1))

    for i in range(iterations):
        y_prediction = np.dot(x_Values, beta)
        error = y_prediction - y_Values
        beta -= learningRate * (1 / n) * np.dot(x_Values.T, error)  # .T means transpose

    return beta

# Set n, k, iteration, learningRate and beta
n = 100
k = 2
beta = np.array([[1], [2], [5]])
iteration = 10000
learningRate = 0.001

print("Iteration:", iteration)
print("Learning rate:", learningRate)

# Generate random values for X
# np.ones((n, 1)) is intercept, and np.random.normal(0, 10, (n, k)) are the independent variables
X = np.hstack([np.ones((n, 1)), np.random.normal(0, 10, (n, k))])


# Generate random values for noise
# (n, 1) generates a numpy array of shape (n, 1) with n rows and 1 column
noise = np.random.normal(0, 1, (n, 1))

# Generate random values for Y
Y = np.dot(X, beta) + noise

# Combine X and Y into a single data array
data = np.hstack([X, Y])


# Prints the result of an estimation of the regression coefficients b0, b1, and b2
print("\nThe estimated coefficients are:", gradientDescentFunction(data, iteration, learningRate).flatten())  # flatten() makes printing in single line
