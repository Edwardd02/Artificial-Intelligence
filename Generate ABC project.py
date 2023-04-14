import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

listTarget = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
input_nodes = 25
hidden_nodes = 10  # overfitting when larger than 20
output_nodes = 3
# Define weights (randomly initialized)
weights_input_hidden = np.random.uniform(0, 1, size=(input_nodes, hidden_nodes))
# Define bias (randomly initialized)
bias_hidden = np.random.uniform(0, 1, size=hidden_nodes)
# Define zs (randomly initialized)
weights_hidden_output = np.random.uniform(0, 1, size=(hidden_nodes, output_nodes))
# Define zBias (randomly initialized)
bias_output = np.random.uniform(0, 1, size=output_nodes)
# Set learning rate and number of iterations
learning_rate = 0.5
num_iterations = 1000
mse_history = []
for n in range(num_iterations):
    meanSquareError = 0
    for i in range(0, len(listX)):
        # Forward propagation
        hiddenLayers = sigmoid(np.dot(listX[i], weights_input_hidden) + bias_hidden)
        output = sigmoid(np.dot(hiddenLayers, weights_hidden_output) + bias_output)

        # Estimate error
        error = listTarget[i] - output
        meanSquareError += np.mean(error ** 2)

        # Backpropagation
        outputDerror = error * sigmoid_derivative(output)
        hiddenDerror = np.dot(outputDerror, weights_hidden_output.T) * sigmoid_derivative(hiddenLayers)

        # Update weights and bias
        weights_hidden_output += np.dot(hiddenLayers.reshape(hidden_nodes, 1),
                                        outputDerror.reshape(1, output_nodes)) * learning_rate
        bias_output += outputDerror * learning_rate
        weights_input_hidden += np.dot(listX[i].reshape(input_nodes, 1),
                                       hiddenDerror.reshape(1, hidden_nodes)) * learning_rate
        bias_hidden += hiddenDerror * learning_rate

    meanSquareError /= len(listX)
    mse_history.append(meanSquareError)

test = np.array([[0, 0, 1, 0, 0,
                  0, 1, 0, 1, 0,
                  0, 1, 1, 1, 0,
                  0, 1, 0, 1, 0,
                  0, 1, 0, 1, 0]])
hiddenLayers = sigmoid(np.dot(test[0], weights_input_hidden) + bias_hidden)
output = sigmoid(np.dot(hiddenLayers, weights_hidden_output) + bias_output)
print(output)
# Plot the mean square error
plt.plot(mse_history)
plt.xlabel('Iteration')
plt.ylabel('Mean Square Error')
plt.title('Mean Square Error vs Iteration')
plt.show()
