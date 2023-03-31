import numpy as np


# 1/e^x
# Define sigmoid function
def sigmoid(x):
    predicted = 1 / (1 + np.exp(-x))
    return predicted


# Define derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


np.random.seed(1)

# Define input data
X1 = 0.2
X2 = 0.3
X3 = 0.5

# Define weights (randomly initialized)
W1 = np.random.uniform(0, 1)
W2 = np.random.uniform(0, 1)
W3 = np.random.uniform(0, 1)

target = 0.5
# Set learning rate and number of iterations
learningRate = 0.3
num_iterations = 100000

# Train the network
for i in range(num_iterations):
    sumOfProducts = X1 * W1 + X2 * W2 + X3 * W3
    error = target - sigmoid(sumOfProducts)
    W1 = W1 + learningRate * error * sigmoid_derivative(sigmoid(sumOfProducts)) * X1
    W2 = W2 + learningRate * error * sigmoid_derivative(sigmoid(sumOfProducts)) * X2
    W3 = W3 + learningRate * error * sigmoid_derivative(sigmoid(sumOfProducts)) * X3

output = sigmoid(X1 * W1 + X2 * W2 + X3 * W3)
print("Target:", target)
print("Output:", output)

# Set random seed for reproducibility
np.random.seed(1)

# Define input data and target output values
listX = [[0, 0], [0, 1], [1, 0], [1, 1]]
lenListX = len(listX)
listTarget = [0, 1, 1, 0]

# Define weights (randomly initialized)
listWeights = [0.1, 0.2, 0.3]
# Set learning rate and number of iterations
learning_rate = 0.3
num_iterations = 10000
error1 = 0
for j in range(num_iterations):
    error = 0
    #error * sigmoid_derivative(sigmoid(sumOfProducts)) * X1
    for i in range(0, lenListX):
        output = sigmoid(listWeights[0] + listWeights[1] * listX[i][0] + listWeights[2] * listX[i][1])
        currentError = listTarget[i] - output
        error += currentError
        gradient = currentError * output * (1 - output)
        listWeights[0] += learning_rate * gradient
        listWeights[1] += learning_rate * gradient * listX[i][0]
        listWeights[2] += learning_rate * gradient * listX[i][1]
print("Target:", listTarget)
print("Weights:", listWeights)
test = [[0, 0], [0, 1], [1, 0], [1, 1]]
output = []
for i in range(0, len(test)):
    output.append(sigmoid(listWeights[0] + listWeights[1] * test[i][0] + listWeights[2] * test[i][1]).round(4))
print("Output:", output)




