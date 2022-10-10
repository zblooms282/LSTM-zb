import numpy as np
from gwu_nn.loss_functions import MSE, LogLoss, CrossEntropy

loss_functions = {'mse': MSE, 'log_loss': LogLoss, 'cross_entropy': CrossEntropy}

class GWUNetwork():
    """The GWUNetwork class is the core class of the library that provies a
    foundation to build a network by iteratively adding layers"""

    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        """A network is comprised of a series of layers connected together. The
        add method provides a means to add a layer to a network
        
        Args:
            Layer (Layer): A Layer object to add to the network
        """
        if len(self.layers) > 0:
            layer.init_weights(self.layers[-1].output_size)
        else:
            layer.init_weights(layer.input_size)
        self.layers.append(layer)

    def get_weights(self):
        pass

    def compile(self, loss, lr):
        """Compile sets a model's loss function and learning rate, preparing the
        model for training
        
        Args:
            loss (LossFunction): The loss function used for the network
            lr (float): The learning rate for the network"""
        if isinstance(loss, str):
            layer_loss = loss_functions[loss]
        else:
            layer_loss = loss
        self.loss = layer_loss.loss
        self.loss_prime = layer_loss.loss_partial_derivative
        self.learning_rate = lr

    # predict output for given input
    def predict(self, input_data):
        """Predict produces predictions for the provided input data
        
        Args:
            input_data (np.array): Input data to inference
        
        Returns:
            np.array: the predictions for the given model
        """
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def evaluate(self, x, y):
        pass

    # train the network
    def fit(self, x_train, y_train, epochs, batch_size=None):
        """Fit is the trianing loop for the model/network
        
        Args:
            x_train (np.array): Inputs for the network to train on
            y_train (np.array): Expected outputs for the network
            epochs (int): Number of training cycles to run through
            batch_size (int): Number of records to train on at a time"""
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j].reshape(1, -1)
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                y_true = np.array(y_train[j]).reshape(-1, 1)
                err += self.loss(y_true, output)

                # backward propagation
                error = self.loss_prime(y_true, output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, self.learning_rate)

            # calculate average error on all samples
            if i % 10 == 0:
                err /= samples
                print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
                
    def __repr__(self):
        rep = "Model:"

        if len(self.layers) < 1:
            return "Model: Empty"
        else:
            rep += "\n"

        for layer in self.layers:
            if layer.type == "Activation":
                rep += f'{layer.name} Activation'
            else:
                rep += f'{layer.name} - ({layer.input_size}, {layer.output_size})\n'

        return rep
