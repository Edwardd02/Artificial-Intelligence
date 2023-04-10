import numpy as np


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
listZs = np.random.uniform(0, 1, size=(50, 3))
# Define zBias (randomly initialized)
zBias = np.random.uniform(0, 1, size=(1, 3))
# Set learning rate and number of iterations
learning_rate = 10
num_iterations = 1000
meanSquareError = 0
for n in range(num_iterations):
    meanSquareError = 0
    for m in range(0, lenListX):
        hiddenLayers = sigmoid(np.add(np.dot(listWeights, listX[m].flatten()), bias))
        output = sigmoid(np.add(np.dot(hiddenLayers, listZs), zBias))
        error = np.sum(listTarget[m] - output)
        meanSquareError += error ** 2
        derror = error * sigmoid_derivative(output)
        dsArray1 = np.dot(hiddenLayers, derror[0][0])
        dsArray2 = np.dot(hiddenLayers, derror[0][1])
        dsArray3 = np.dot(hiddenLayers, derror[0][2])
        dhArray = (np.dot(listZs, derror.flatten())) * sigmoid_derivative(hiddenLayers)
        zBias = np.add(zBias, learning_rate * derror)
        for i in range(len(listZs)):
            listZs[i][0] = np.add(listZs[i][0], learning_rate * dsArray1[0][i])
            listZs[i][1] = np.add(listZs[i][1], learning_rate * dsArray2[0][i])
            listZs[i][2] = np.add(listZs[i][2], learning_rate * dsArray3[0][i])
        bias = np.add(bias, learning_rate * dhArray)
        for i in range(len(listWeights)):
            for j in range(len(listWeights[i])):
                listWeights[i][j] = np.add(listWeights[i][j], learning_rate * dhArray[0][i] * listX[0][j])
print("Mean Square Error:", meanSquareError)
test = np.array([[0, 0, 1, 0, 0,
                    0, 1, 0, 1, 0,
                    0, 1, 1, 1, 0,
                    0, 1, 0, 1, 0,
                    0, 1, 0, 1, 0]])
hiddenLayers = sigmoid(np.add(np.dot(listWeights, test.flatten()), bias))
output = sigmoid(np.add(np.dot(hiddenLayers, listZs), zBias))
print(output)
