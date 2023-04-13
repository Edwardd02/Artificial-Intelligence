import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


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

# Define the architecture
input_nodes = 25
hidden_nodes = 18  # overfitting when larger than 20
output_nodes = 3

# Initialize weights and biases
weights_input_hidden = np.random.uniform(0, 1, size=(input_nodes, hidden_nodes))
weights_hidden_output = np.random.uniform(0, 1, size=(hidden_nodes, output_nodes))
bias_hidden = np.random.uniform(0, 1, size=hidden_nodes)
bias_output = np.random.uniform(0, 1, size=output_nodes)

learning_rate = 3
num_iterations = 1000
mse_history = []
for j in range(num_iterations):
    meanSquareError = 0
    for i in range(len(listX)):
        # Forward propagation
        hidden_layer_input = np.dot(listX[i], weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        output_layer_output = sigmoid(output_layer_input)

        # Calculate error
        error = listTarget[i] - output_layer_output
        meanSquareError += np.mean(error ** 2)

        # Backpropagation
        derror_output = error * sigmoid_derivative(output_layer_output)
        derror_hidden = np.dot(derror_output, weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

        # Update weights and biases
        weights_hidden_output += learning_rate * np.outer(hidden_layer_output, derror_output)
        weights_input_hidden += learning_rate * np.outer(listX[i], derror_hidden)
        bias_output += learning_rate * derror_output
        bias_hidden += learning_rate * derror_hidden

    mse_history.append(meanSquareError / len(listX))
print("Mean Square Error:", mse_history[-1])
print("Target:", listTarget)

finalOutput = []

for i in range(listX.shape[0]):
    hidden_layer_input = np.dot(listX[i], weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)
    finalOutput.append(np.round(output_layer_output))

print("Output:", finalOutput)

# Plot the mean square error
plt.plot(mse_history)
plt.xlabel('Iteration')
plt.ylabel('Mean Square Error')
plt.title('Mean Square Error vs Iteration')
plt.show()
