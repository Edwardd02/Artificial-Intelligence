import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    predicted = 1 / (1 + np.exp(-x))
    return predicted


def sigmoid_derivative(x):
    return x * (1 - x)


# Set random seed for reproducibility

# Define input data and target output values
listX = np.array([[0, 0, 1, 0, 0,
                   0, 1, 0, 1, 0,
                   0, 1, 1, 1, 0,
                   0, 1, 0, 1, 0,
                   0, 1, 0, 1, 0],
                  [0, 1, 1, 0, 0,
                   0, 1, 0, 1, 0,
                   0, 1, 1, 0, 0,
                   0, 1, 0, 1, 0,
                   0, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0,
                   1, 0, 0, 0, 0,
                   1, 0, 0, 0, 0,
                   1, 0, 0, 0, 0,
                   0, 1, 1, 1, 0]])
lenListX = len(listX)
listTarget = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
# Define weights (randomly initialized)
listWeights = np.random.uniform(0, 1, size=(50, 25))
# Define bias (randomly initialized)
bias = np.random.uniform(0, 1, size=(1, 50))
# Define zs (randomly initialized)
listZs = np.random.uniform(0, 1, size=(3, 50))
# Define zBias (randomly initialized)
zBias = np.random.uniform(0, 1, size=(1, 3))
# Set learning rate and number of iterations
learning_rate = 10
num_iterations = 1000
meanSquareError = 0
mse_history = []
for m in range(0, lenListX):
    hiddenLayers = np.array([])
    for i in range(len(listWeights)):
        sum = 0
        for j in range(len(listWeights[i])):
            sum += listWeights[i][j] * listX[m][j]
        sum += bias[0][i]
        hiddenLayers = np.append(hiddenLayers, sigmoid(sum))

    output = np.array([])
    for i in range(len(listZs)):
        sum = 0
        for j in range(len(listZs[i])):
            sum += listZs[i][j] * hiddenLayers[j]
        sum += zBias[0][i]
        output = np.append(output, sigmoid(sum))

    error = listTarget[m] - output
    meanSquareError += error ** 2
    derror = error * sigmoid_derivative(output)
    dsArray1 = np.array([])
    dsArray2 = np.array([])
    dsArray3 = np.array([])
    for i in range (len(hiddenLayers)):
        dsArray1 = np.append(dsArray1, derror[0] * listZs[0][i])
        dsArray2 = np.append(dsArray2, derror[1] * listZs[1][i])
        dsArray3 = np.append(dsArray3, derror[2] * listZs[2][i])
    dhArray = np.array([])
    for i in range(len(listWeights)):
        dhArray = np.append(dhArray, derror[0] * derror[1] * derror[2] * sigmoid_derivative(hiddenLayers[i]) * listZs[0][i] * listZs[1][i] * listZs[2][i])
    for i in range(len(zBias)):
        zBias[i] = np.add(zBias[i], learning_rate * derror[i])
    for i in range(len(listZs[0])):
        listZs[0][i] = np.add(listZs[0][i], learning_rate * dsArray1[i])
        listZs[1][i] = np.add(listZs[1][i], learning_rate * dsArray2[i])
        listZs[2][i] = np.add(listZs[2][i], learning_rate * dsArray3[i])
    for i in range(len(bias)):
        bias = np.add(bias, learning_rate * dhArray[i])
    for i in range(len(listWeights)):
        for j in range(len(listWeights[i])):
            listWeights[i][j] = np.add(listWeights[i][j], learning_rate * dhArray[i] * listX[m][j])
print("Mean Square Error:", meanSquareError)
test = np.array([[0, 0, 1, 0, 0,
                  0, 1, 0, 1, 0,
                  0, 1, 1, 1, 0,
                  0, 1, 0, 1, 0,
                  0, 1, 0, 1, 0]])
hiddenLayers = sigmoid(np.add(np.dot(listWeights, test[0]), bias))
output = sigmoid(np.add(np.dot(hiddenLayers, listZs), zBias))
print(output)
