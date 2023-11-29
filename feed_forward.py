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

    def forward_propagation(self, X):
        self.hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        self.hidden_outputs = self.activation_function(self.hidden_inputs)

        self.final_inputs = np.dot(self.hidden_outputs, self.weights_hidden_to_output)
        self.final_outputs = self.activation_function(self.final_inputs)
        return self.final_outputs

    def backward_propagation(self, X, y, final_outputs):
        output_errors = y - final_outputs
        hidden_errors = np.dot(output_errors, self.weights_hidden_to_output.T)

        self.weights_hidden_to_output += self.lr * np.dot(self.hidden_outputs.T, output_errors)
        self.weights_input_to_hidden += self.lr * np.dot(X.T, hidden_errors)
