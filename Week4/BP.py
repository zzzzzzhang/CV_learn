# coding: utf-8

import numpy as np

class NeuralNetwork():
#     a three-layer BP neural network
#     init function
    def __init__(self, n_input, n_hidden, n_output, lr = 0.01):
        self.input = np.mat(np.ones(n_input + 1))
        self.w_input = np.mat(np.ones((n_input + 1,n_hidden)))
        self.hidden = np.mat(np.ones(n_hidden + 1))
        self.w_hidden = np.mat(np.ones((n_hidden +1,n_output)))
        self.output = np.mat(np.ones(n_output))
        self.activation = 'sigmoid'
        self.lr = lr
#     activation
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    
#     calculate
    def get_data(self, inputs,outputs):
        assert len(inputs)+1 == self.input.shape[1]
        assert len(outputs) == self.output.shape[1]
        self.input[:,:-1] = np.mat(inputs)
        self.labels = np.mat(outputs)
        
    def forward_Calculate(self):
        self.hidden[:,:-1] = sigmoid(self.input * self.w_input)
        self.output = self.hidden * self.w_hidden
#     calculate loss        
        self.loss = ((self.output - self.labels).A**2).sum()

nn = NeuralNetwork(2,3,2)
nn.get_data([1,1], [2,0])
nn.forward_Calculate()

