import math
import random


def sigmoid(x):
    predicted = 1 / (1 + math.e ** (-x))
    return predicted
def sigmoid_derivative(x):
    return x * (1 - x)
# derror = (target - output) * sigmoid_derivative(output)
# ds1 = derror * sigmoid_derivative(h1) * X1
# ds2 = derror * sigmoid_derivative(h1) * X2
# ds3 = derror * sigmoid_derivative(h1) * X3

# Set random seed for reproducibility

# Define input data and target output values
listX = [[0, 0], [0, 1], [1, 0], [1, 1]]
lenListX = len(listX)
listTarget = [0, 1, 1, 0]

# Define weights (randomly initialized)
listWeights = [[random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)],
               [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]]
listZs = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
bais = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
baisZ = random.uniform(0, 1)
# Set learning rate and number of iterations
learning_rate = 10
num_iterations = 1000
for j in range(num_iterations):
    # error * sigmoid_derivative(sigmoid(sumOfProducts)) * X1
    meanSquareError = 0
    for i in range(0, lenListX):
        h1 = sigmoid(bais[0] + listWeights[0][0] * listX[i][0] + listWeights[1][0] * listX[i][1])
        h2 = sigmoid(bais[1] + listWeights[0][1] * listX[i][0] + listWeights[1][1] * listX[i][1])
        h3 = sigmoid(bais[2] + listWeights[0][2] * listX[i][0] + listWeights[1][2] * listX[i][1])

        output = sigmoid(baisZ + listZs[0] * h1 + listZs[1] * h2 + listZs[2] * h3)
        error = listTarget[i] - output

        derror = error * sigmoid_derivative(output)
        meanSquareError += error ** 2
        ds1 = derror * h1
        ds2 = derror * h2
        ds3 = derror * h3
        dh1 = derror * sigmoid_derivative(h1) * listZs[0]
        dh2 = derror * sigmoid_derivative(h2) * listZs[1]
        dh3 = derror * sigmoid_derivative(h3) * listZs[2]
        baisZ += learning_rate * derror
        listZs[0] += learning_rate * ds1
        listZs[1] += learning_rate * ds2
        listZs[2] += learning_rate * ds3
        #gradient = currentError * output * (1 - output)
        bais[0] += learning_rate * dh1
        bais[1] += learning_rate * dh2
        bais[2] += learning_rate * dh3
        listWeights[0][0] += learning_rate * dh1 * listX[i][0]
        listWeights[0][1] += learning_rate * dh2 * listX[i][0]
        listWeights[0][2] += learning_rate * dh3 * listX[i][0]
        listWeights[1][0] += learning_rate * dh1 * listX[i][1]
        listWeights[1][1] += learning_rate * dh2 * listX[i][1]
        listWeights[1][2] += learning_rate * dh3 * listX[i][1]

print("Mean Square Error:", meanSquareError)
print("Target:", listTarget)
print("Weights:", listWeights)
test = [[0, 0], [0, 1], [1, 0], [1, 1]]
finalOutput = []
for i in range(0, len(test)):
    h1 = sigmoid(bais[0] + listWeights[0][0] * test[i][0] + listWeights[1][0] * test[i][1])
    h2 = sigmoid(bais[1] + listWeights[0][1] * test[i][0] + listWeights[1][1] * test[i][1])
    h3 = sigmoid(bais[2] + listWeights[0][2] * test[i][0] + listWeights[1][2] * test[i][1])
    output = sigmoid(baisZ + listZs[0] * h1 + listZs[1] * h2 + listZs[2] * h3)
    finalOutput.append(output)
print("Output:", finalOutput)
