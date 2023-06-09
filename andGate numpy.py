import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    predicted = 1 / (1 + np.exp(-x))
    return predicted


# Set random seed for reproducibility

# Define input data and target output values
listX = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
lenListX = len(listX)
listTarget = np.array([0, 0, 0, 1])

# Define weights (randomly initialized)
listWeights = np.array([np.random.uniform(0, 1), np.random.uniform(0, 1)])
bias = np.random.uniform(0, 1)
# Set learning rate and number of iterations
learning_rate = 10
num_iterations = 10000
mse_history = []
for j in range(num_iterations):
    # error * sigmoid_derivative(sigmoid(sumOfProducts)) * X1
    meanSquareError = 0
    for i in range(0, lenListX):
        output = sigmoid(np.sum(np.dot(listWeights, listX[i])) + bias)
        currentError = listTarget[i] - output
        meanSquareError += currentError ** 2
        gradient = currentError * output * (1 - output)
        bias += learning_rate * gradient
        listWeights[0] += learning_rate * gradient * listX[i][0]
        listWeights[1] += learning_rate * gradient * listX[i][1]
    mse_history.append(meanSquareError / lenListX)
print("Target:", listTarget)
print("Weights:", listWeights)
test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output = np.array([])
for i in range(0, len(test)):
    output = np.append(output, sigmoid(np.sum(np.dot(listWeights, test[i])) + bias))
print("Output:", output)
plt.plot(mse_history)
plt.xlabel('Iteration')
plt.ylabel('Mean Square Error')
plt.title('Mean Square Error vs Iteration')
plt.show()