import numpy as np

class Perceptron:
    # how many perceptron - num_inputs
    # learning rate - in backpropagation, learning rate determines by how much to change the weights, 
    # lower learning rate means smaller model will learn
    # epochs - how many training cycles
    # weights - multiplication factors for each input

    def __init__(self, num_inputs, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(num_inputs) # initialize weights to zero
        self.bias = 0 # adjustment factor
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def predict(self, inputs):
        # calculate weighted sum
        summation = np.dot(self.weights, inputs) + self.bias # there is a formula for this

        # binary classification
        return 1 if summation > 0 else 0 # activation function (step function) not sigmoid
    
    def train(self, training_inputs, labels):
        # epoch - one complete pass through the training dataset
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)

                # update weights and bias based on error
                error = label - prediction
                # check how much is diff between predicted and actual
                # so we have to adjust weights and bias accordingly per epoch
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error