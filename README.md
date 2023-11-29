### HOW TO CREATE

Creating a simple feed-forward neural network in Python involves several steps. We'll use the numpy library for matrix operations and the scikit-learn library for data manipulation.

First, let's define the structure of our neural network. We'll create a class called FeedForwardNeuralNetwork that will contain methods for initialization, forward propagation, and backward propagation.

import numpy as np

class FeedForwardNeuralNetwork:
   def __init__(self, input_nodes, hidden_nodes, output_nodes):
       self.input_nodes = input_nodes
       self.hidden_nodes = hidden_nodes
       self.output_nodes = output_nodes

       self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                    (self.input_nodes, self.hidden_nodes))

       self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                    (self.hidden_nodes, self.output_nodes))
       self.lr = 0.1
       
       self.activation_function = lambda x : 1 / (1 + np.exp(-x)) # sigmoid function
4

In the __init__ method, we initialize the weights for the connections between the input layer and the hidden layer, and between the hidden layer and the output layer. The weights are initialized with random values. The learning rate is also set to 0.1.

The activation_function is set to the sigmoid function, but you can use any other activation function such as ReLU, tanh, etc.

Next, let's implement the forward propagation method. This method takes an input and passes it through the network to get an output.

   def forward_propagation(self, X):
       self.hidden_inputs = np.dot(X, self.weights_input_to_hidden)
       self.hidden_outputs = self.activation_function(self.hidden_inputs)

       self.final_inputs = np.dot(self.hidden_outputs, self.weights_hidden_to_output)
       self.final_outputs = self.activation_function(self.final_inputs)
4

In the forward_propagation method, we first calculate the hidden layer inputs by multiplying the input with the weights from the input layer to the hidden layer. We then pass these inputs through the activation function to get the hidden layer outputs.

Next, we calculate the final layer inputs by multiplying the hidden layer outputs with the weights from the hidden layer to the output layer. We then pass these inputs through the activation function to get the final outputs.

Finally, let's implement the backward propagation method. This method adjusts the weights based on the error between the predicted output and the actual output.

   def backward_propagation(self, X, y, final_outputs):
       output_errors = y - final_outputs
       hidden_errors = np.dot(output_errors, self.weights_hidden_to_output.T)

       self.weights_hidden_to_output += self.lr * np.dot(self.hidden_outputs.T, output_errors)
       self.weights_input_to_hidden += self.lr * np.dot(X.T, hidden_errors)


In the backward_propagation method, we first calculate the output errors by subtracting the actual output from the predicted output. We then calculate the hidden layer errors by multiplying the output errors with the transpose of the weights from the hidden layer to the output layer.

Next, we adjust the weights from the hidden layer to the output layer by adding the product of the learning rate and the dot product of the transpose of the hidden layer outputs and the output errors. We do the same for the weights from the input layer to the hidden layer.

This is a very basic implementation of a feed-forward neural network in Python. Depending on your specific needs, you might want to add more features such as bias nodes, different activation functions, etc.

SOURCES
favicon-hackernoon.com
hackernoon.com > building-a-feedforward-neural-network-from-scratch-in-python-d3526457156b
Building a Feedforward Neural Network from Scratch in Python
Feedforward neural networks are also known as Multi-layered Network of Neurons (MLN) These network of models are called feedforward because the information only..

favicon-pub.towardsai.net
pub.towardsai.net > building-feedforward-neural-networks-from-scratch-c5a5cf23b97b
Building Feedforward Neural Networks from Scratch
Nicolò Tognoni · Follow Published in Towards AI · 12 min read · Oct 1, 2022 1 Photo by Uriel SC, on Unsplash This article..

favicon-vitalflux.com
vitalflux.com > feed-forward-neural-network-python-example
Feedforward Neural Network Python Example - Analytics Yogi
June 11, 2022 by Ajitesh Kumar · 1 Comment A feedforward neural network, also known as a multi-layer perceptron, is composed of layers of neurons..

favicon-medium.com
medium.com > hackernoon > building-a-feedforward-neural-network-from-scratch-in-python-d3526457156b
Building a Feedforward Neural Network from Scratch in Python
Building a Feedforward Neural Network from Scratch in Python Niranjan Kumar · Follow Published in HackerNoon.com · 18 min read · Apr 9, 2019 --..

favicon-www.section.io
www.section.io > engineering-education > feedforward-and-recurrent-neural-networks-python-implementation
Feed-forward and Recurrent Neural Networks Python ... - Section
Import libraries Create sample weights Include weights Forward propagation of the input signal Applications of Feed-forward neural network Recurrent neural netw..


### HOW TO USE

Instantiate the Neural Network: You first need to create an instance of the FeedForwardNeuralNetwork class. You need to specify the number of input nodes, hidden nodes, and output nodes.
network = FeedForwardNeuralNetwork(input_nodes=2, hidden_nodes=2, output_nodes=1)
Train the Network: Next, you need to train the network. This involves running the forward propagation method and the backward propagation method multiple times. In each iteration, you should provide the network with some input data and the corresponding output data. The network will then adjust its weights based on the difference between its output and the actual output.
for i in range(1500):
   # Forward propagation
   network.forward_propagation(X)
   # Backward propagation
   network.backward_propagation(X, y, network.final_outputs)
Predict: Once the network is trained, you can use it to make predictions on new data. This involves running the forward propagation method with the new data and then returning the final outputs.
predictions = network.forward_propagation(new_data)
Here is a complete example of how to use the feed-forward neural network:

import numpy as np

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
In this example, we first define the neural network and generate some training data. We then train the network by running the forward propagation method and the backward propagation method multiple times. Finally, we generate some new data and use the network to make predictions on this data.

Please note that this is a very basic example and the actual usage of the neural network might be different depending on your specific needs and the structure of your data 1.

SOURCES
favicon-hackernoon.com
hackernoon.com > building-a-feedforward-neural-network-from-scratch-in-python-d3526457156b
Building a Feedforward Neural Network from Scratch in Python
Feedforward neural networks are also known as Multi-layered Network of Neurons (MLN) These network of models are called feedforward because the information only..

favicon-pyimagesearch.com
pyimagesearch.com > 2021 > 05 > 06 > implementing-feedforward-neural-networks-with-keras-and-tensorflow
Implementing feedforward neural networks with Keras and ... - PyImageSearch
Implementing feedforward neural networks with Keras and TensorFlow by Adrian Rosebrock on May 6, 2021 Click here to download the source code to this post..

favicon-vitalflux.com
vitalflux.com > feed-forward-neural-network-python-example
Feedforward Neural Network Python Example - Analytics Yogi
June 11, 2022 by Ajitesh Kumar · 1 Comment A feedforward neural network, also known as a multi-layer perceptron, is composed of layers of neurons..

favicon-www.section.io
www.section.io > engineering-education > feedforward-and-recurrent-neural-networks-python-implementation
Feed-forward and Recurrent Neural Networks Python ... - Section
Create sample weights Include weights Forward propagation of the input signal Applications of Feed-forward neural network Recurrent neural network RNN implement..

favicon-www.deeplearningwizard.com
www.deeplearningwizard.com > deep_learning > practical_pytorch > pytorch_feedforward_neuralnetwork
Feedforward Neural Networks (FNN) - Deep Learning Wizard
Instantiate the logistic regression model. input_dim = 28*28 output_dim = 10 model = LogisticRegressionModel(input_dim, output_dim) When we inspect the model, w..
