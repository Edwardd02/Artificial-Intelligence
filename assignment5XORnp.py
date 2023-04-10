import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


listX = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
listTarget = np.array([0, 1, 1, 0])

listWeights = np.random.rand(2, 3)  # Hidden layer with three nodes
listZs = np.random.rand(3)  # Output layer with one node
bias = np.random.rand(3)
biasZ = np.random.rand()

learning_rate = 0.5
num_iterations = 5000
mse_history = []

for j in range(num_iterations):
    meanSquareError = 0
    for i in range(listX.shape[0]):
        h = sigmoid(bias + listX[i] @ listWeights)
        output = sigmoid(biasZ + h @ listZs)
        error = listTarget[i] - output
        derror = error * sigmoid_derivative(output)
        meanSquareError += error ** 2 / listX.shape[0]

        ds = derror * h
        dh = derror * sigmoid_derivative(h) * listZs
        biasZ += learning_rate * derror
        listZs += learning_rate * ds
        bias += learning_rate * dh
        listWeights += learning_rate * np.outer(listX[i], dh)
    mse_history.append(meanSquareError)

print("Mean Square Error:", mse_history[-1])
print("Target:", listTarget)
print("Weights:", listWeights)

finalOutput = []

for i in range(listX.shape[0]):
    h = sigmoid(bias + listX[i] @ listWeights)
    output = sigmoid(biasZ + h @ listZs)
    finalOutput.append(output)

print("Output:", finalOutput)

# Plot the mean square error
plt.plot(mse_history)
plt.xlabel('Iteration')
plt.ylabel('Mean Square Error')
plt.title('Mean Square Error vs Iteration')
plt.show()
