import random

"""
ASSIGNMENT3
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
gradientDescentFunction(data,iterations,learningRate).
"""

# Function to add matrices
def add_matrices(arr1, arr2):
    if ((len(arr1) != len(arr2)) or (len(arr1[0]) != len(arr2[0]))):
        raise ValueError("The columns or the row doesn't match")
    for i in range(len(arr1)):
        for j in range(len(arr1[0])):
            arr1[i][j] = arr1[i][j] + arr2[i][j]
    return arr1


# Function to multiply matrices
def multi_matrices(arr1, arr2):
    arr_result = []
    k = 0
    if (len(arr1[0]) != len(arr2)):
        raise ValueError("The columns of first arr doesn't equals to the rows of second arr")

    for i in range(len(arr1)):
        column_result = []
        for j in range(len(arr2[0])):
            sum = 0
            for k in range(len(arr1[0])):
                sum += arr1[i][k] * arr2[k][j]
            column_result.append(sum)
        arr_result.append(column_result)

    return arr_result


def gradientDescentFunction(data, iteration, learningRate):
    currentError = 0
    count = 0
    x_Values = []
    y_Values = []
    for element in data:
        y_Rows = []
        y_Rows.append(element.pop())
        x_Values.append(element)
        y_Values.append(y_Rows)
    #print(x_Values)
    b0 = 0
    b1 = 0
    b2 = 0
    while (count < iteration):
        prediction_Y = []
        for i in range(0, len(data)):
            prediction_Yrow = []
            prediction_Yrow.append(b0 + b1 * x_Values[i][1] + b2 * x_Values[i][2])
            prediction_Y.append(prediction_Yrow)

        gradientB0 = 0
        gradientB1 = 0
        gradientB2 = 0

        for j in range(0, len(data)):
            gradientB0 = gradientB0 + (y_Values[j][0] - b0 - b1 * x_Values[j][1] - b2 * x_Values[j][2])
            gradientB1 = gradientB1 + (y_Values[j][0] - b0 - b1 * x_Values[j][1] - b2 * x_Values[j][2]) * x_Values[j][1]
            gradientB2 = gradientB2 + (y_Values[j][0] - b0 - b1 * x_Values[j][1] - b2 * x_Values[j][2]) * x_Values[j][2]

        b0 = b0 - learningRate * (-2) * (1 / len(data)) * gradientB0
        b1 = b1 - learningRate * (-2) * (1 / len(data)) * gradientB1
        b2 = b2 - learningRate * (-2) * (1 / len(data)) * gradientB2

        previousError = currentError

        # If the error is less than 0.01, it will stop the iteration
        # Otherwise, the process continues until the inputted iteration value
        for z in range(0, len(data)):
            currentError = currentError + (prediction_Y[z][0] - y_Values[z][0]) ** 2
        count += 1
        if (abs(previousError - currentError) < 0.01):
            break

    print("\nActual iteration:", count)
    return (b0, b1, b2)

# Set n, k, iteration, learningRate and beta
n = 100
k = 2
beta = [[1], [2], [5]]
iteration = 10000
learningRate = 0.001
print("Inputted iteration:", iteration)
print("Learning rate:", learningRate)

x_Values = []
noise = []

# Generate random values for X
for i in range(0, n):
    xRows = [1]
    for j in range(0, k):
        xRows.append(random.normalvariate(0, 1))  # 0 is mean, 10 is standard deviation here
    x_Values.append(xRows)

# Generate random values for noise
for i in range(0, n):
    noiseRows = []
    noiseRows.append(random.normalvariate(0, 1))
    noise.append(noiseRows)

# Generates Y values and Print the random datas
y_Values = add_matrices(multi_matrices(x_Values, beta), noise)

"""
The part below prints a lot for debugging
Please remove or comment out if necessary
"""

# # Prints random data of the form yₙ = β0 + β₀ + β₁xₙ₁ + β₂xₙ₂ + ⋯ + βₖxₙₖ + εₙ
# for i in range(0, len(x_Values)):
#     print("Y" + str(i + 1), "=", beta[0][0], "+", beta[1][0], "*", x_Values[i][1], "+", beta[2][0], "*", x_Values[i][2],
#           "+", noise[i][0])
#
# # Prints the sum for every form
# for i in range(0, len(y_Values)):
#     print("Y" + str(i + 1), "=", y_Values[i][0])

# Combine X and Y Values
for i in range(0, len(x_Values)):
    x_Values[i].append(y_Values[i][0])

data = x_Values
#print(data)


# Prints the result of an estimation of the regression coefficients b0, b1, and b2
print("\nThe estimated coefficients are:", gradientDescentFunction(data, iteration, learningRate))
