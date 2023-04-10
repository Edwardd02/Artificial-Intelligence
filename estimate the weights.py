import random
import math
import matplotlib.pyplot as plt
# 1/e^x
# Define sigmoid function
def sigmoid(x):
    predicted = 1 / (1 + math.exp(-x))
    return predicted


# Define derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)



# Define input data
X1 = 0.2
X2 = 0.3
X3 = 0.5

# Define weights (randomly initialized)
W1 = random.uniform(0, 1)
W2 = random.uniform(0, 1)
W3 = random.uniform(0, 1)

target = 0.5
# Set learning rate and number of iterations
learningRate = 0.3
num_iterations = 100000
mse_history = []
# Train the network
for i in range(num_iterations):
    sumOfProducts = X1 * W1 + X2 * W2 + X3 * W3
    error = target - sigmoid(sumOfProducts)
    mse_history.append(error ** 2)
    W1 = W1 + learningRate * error * sigmoid_derivative(sigmoid(sumOfProducts)) * X1
    W2 = W2 + learningRate * error * sigmoid_derivative(sigmoid(sumOfProducts)) * X2
    W3 = W3 + learningRate * error * sigmoid_derivative(sigmoid(sumOfProducts)) * X3

output = sigmoid(X1 * W1 + X2 * W2 + X3 * W3)
print("Target:", target)
print("Output:", output)
plt.plot(mse_history)
plt.xlabel('Iteration')
plt.ylabel('Mean Square Error')
plt.title('Mean Square Error vs Iteration Estimate the weights')
plt.show()







