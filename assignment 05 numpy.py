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
inputs = np.array([0.2, 0.3, 0.5])

# Define weights (randomly initialized)
weights = np.array([np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)])

target = 0.5
# Set learning rate and number of iterations
learningRate = 0.3
num_iterations = 100000

# Train the network
for i in range(num_iterations):
    sumOfProducts = np.sum(np.dot(inputs, weights))
    error = target - sigmoid(sumOfProducts)
    weights[0] += learningRate * error * sigmoid_derivative(sigmoid(sumOfProducts)) * inputs[0]
    weights[1] += learningRate * error * sigmoid_derivative(sigmoid(sumOfProducts)) * inputs[1]
    weights[2] += learningRate * error * sigmoid_derivative(sigmoid(sumOfProducts)) * inputs[2]

output = sigmoid(np.dot(inputs, weights))
print("Target:", target)
print("Output:", output)





