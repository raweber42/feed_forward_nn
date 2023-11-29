import numpy as np

from feed_forward import FeedForwardNeuralNetwork

# Define the neural network
network = FeedForwardNeuralNetwork(input_nodes=2, hidden_nodes=2, output_nodes=1)

# Generate some training data
X = np.random.rand(100, 2)
y = np.random.rand(100, 1)

# Train the network
for i in range(1500):
   network.forward_propagation(X)
   network.backward_propagation(X, y, network.final_outputs)

# Generate some new data
new_data = np.random.rand(10, 2)

# Make predictions
predictions = network.forward_propagation(new_data)

print(predictions)
