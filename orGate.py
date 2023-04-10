import math
import random
import matplotlib.pyplot as plt


def sigmoid(x):
    predicted = 1 / (1 + math.exp(-x))
    return predicted


# Set random seed for reproducibility

# Define input data and target output values
listX = [[0, 0], [0, 1], [1, 0], [1, 1]]
lenListX = len(listX)
listTarget = [0, 1, 1, 1]

# Define weights (randomly initialized)
listWeights = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
# Set learning rate and number of iterations
learning_rate = 10
num_iterations = 10000
mse_history = []
for j in range(num_iterations):
    # error * sigmoid_derivative(sigmoid(sumOfProducts)) * X1
    meanSquareError = 0
    for i in range(0, lenListX):
        output = sigmoid(listWeights[0] + listWeights[1] * listX[i][0] + listWeights[2] * listX[i][1])
        currentError = listTarget[i] - output
        meanSquareError += currentError ** 2
        gradient = currentError * output * (1 - output)
        listWeights[0] += learning_rate * gradient
        listWeights[1] += learning_rate * gradient * listX[i][0]
        listWeights[2] += learning_rate * gradient * listX[i][1]
    mse_history.append(meanSquareError)
print("Target:", listTarget)
print("Weights:", listWeights)
test = [[0, 0], [0, 1], [1, 0], [1, 1]]
output = []
for i in range(0, len(test)):
    output.append(sigmoid(listWeights[0] + listWeights[1] * test[i][0] + listWeights[2] * test[i][1]))
print("Output:", output)
plt.plot(mse_history)
plt.xlabel('Iteration')
plt.ylabel('Mean Square Error')
plt.title('Mean Square Error vs Iteration')
plt.show()

