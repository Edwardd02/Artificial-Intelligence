import math
import random
import matplotlib.pyplot as plt

def sigmoid(x):
    predicted = 1 / (1 + math.e ** (-x))
    return predicted

def sigmoid_derivative(x):
    return x * (1 - x)

listX = [[0, 0], [0, 1], [1, 0], [1, 1]]
lenListX = len(listX)
listTarget = [0, 1, 1, 0]

# Hidden layer with three nodes
listWeights = [[random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)],
               [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]]
# Output layer with one node
listZs = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
bias = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
biasZ = random.uniform(0, 1)

learning_rate = 0.5
num_iterations = 5000
mse_history = []

for j in range(num_iterations):
    meanSquareError = 0
    for i in range(0, lenListX):
        h1 = sigmoid(bias[0] + listWeights[0][0] * listX[i][0] + listWeights[1][0] * listX[i][1])
        h2 = sigmoid(bias[1] + listWeights[0][1] * listX[i][0] + listWeights[1][1] * listX[i][1])
        h3 = sigmoid(bias[2] + listWeights[0][2] * listX[i][0] + listWeights[1][2] * listX[i][1])

        output = sigmoid(biasZ + listZs[0] * h1 + listZs[1] * h2 + listZs[2] * h3)
        error = listTarget[i] - output

        derror = error * sigmoid_derivative(output)
        meanSquareError += error ** 2 / lenListX
        ds1 = derror * h1
        ds2 = derror * h2
        ds3 = derror * h3
        dh1 = derror * sigmoid_derivative(h1) * listZs[0]
        dh2 = derror * sigmoid_derivative(h2) * listZs[1]
        dh3 = derror * sigmoid_derivative(h3) * listZs[2]
        biasZ += learning_rate * derror
        listZs[0] += learning_rate * ds1
        listZs[1] += learning_rate * ds2
        listZs[2] += learning_rate * ds3
        bias[0] += learning_rate * dh1
        bias[1] += learning_rate * dh2
        bias[2] += learning_rate * dh3
        listWeights[0][0] += learning_rate * dh1 * listX[i][0]
        listWeights[0][1] += learning_rate * dh2 * listX[i][0]
        listWeights[0][2] += learning_rate * dh3 * listX[i][0]
        listWeights[1][0] += learning_rate * dh1 * listX[i][1]
        listWeights[1][1] += learning_rate * dh2 * listX[i][1]
        listWeights[1][2] += learning_rate * dh3 * listX[i][1]
    mse_history.append(meanSquareError)

print("Mean Square Error:", mse_history[-1])
print("Target:", listTarget)
print("Weights:", listWeights)

finalOutput = []

for i in range(0, len(listX)):
    h1 = sigmoid(bias[0] + listWeights[0][0] * listX[i][0] + listWeights[1][0] * listX[i][1])
    h2 = sigmoid(bias[1] + listWeights[0][1] * listX[i][0] + listWeights[1][1] * listX[i][1])
    h3 = sigmoid(bias[2] + listWeights[0][2] * listX[i][0] + listWeights[1][2] * listX[i][1])
    output = sigmoid(biasZ + listZs[0] * h1 + listZs[1] * h2 + listZs[2] * h3)
    finalOutput.append(output)

print("Output:", finalOutput)

# Plot the mean square error
plt.plot(mse_history)
plt.xlabel('Iteration')
plt.ylabel('Mean Square Error')
plt.title('Mean Square Error vs Iteration')
plt.show()