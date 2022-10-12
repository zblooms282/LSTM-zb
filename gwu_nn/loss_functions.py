import numpy as np
from gwu_nn.activation_functions import SoftmaxActivation
from abc import ABC, abstractmethod


class LossFunction(ABC):
    """Abstract Class for loss functions"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(cls, y_true, y_pred):
        """Calculates the loss for the given network.

        Args:
            y_true (np.array): the true values
            y_pred (np.array): the predicted values

        Returns:
            np.array(np.array): the network's loss
        """
        pass

    @abstractmethod
    def loss_partial_derivative(cls, y_true, y_pred):
        """Calculates the derivative of the loss for the given network.
        
        Args:
            y_true (np.array): the true values
            y_pred (np.array): the predicted values

        Returns:
            np.array(np.array): the partial derivative for the network's loss
        """
        pass


class MSE(LossFunction):
    """Class for implementing the MSE loss function. Inheirits
    loss and loss_partial_derivative from LossFunction"""
    @classmethod
    def loss(cls, y_true, y_pred):
        """Calculates the MSE for the true vs predicted values
        
        Returns:
            np.array: MSE for each input
        """
        y_pred = y_pred.reshape(-1)
        return np.mean(np.power(y_true - y_pred, 2))

    @classmethod
    def loss_partial_derivative(cls, y_true, y_pred):
        """Calculates the derivative of the MSE for each prediction
        
        Returns:
            np.array: Partial derivative of the MSE
        """
        y_pred = y_pred.reshape(-1)
        grad = 2 * (y_pred - y_true) / y_pred.size
        grad = grad.reshape(-1, 1)
        return grad

class LogLoss(LossFunction):
    """Class for implementing the LogLoss loss function. Inheirits
    loss and loss_partial_derivative from LossFunction
    """
    @classmethod
    def loss(cls, y_true, y_pred):
        """Calculates the LogLoss for the true vs predicted values
        
        Returns:
            np.array: LogLoss for each input
        """
        return np.mean(-np.log(y_pred)*y_true + -np.log(1-y_pred)*(1-y_true))

    @classmethod
    def loss_partial_derivative(cls, y_true, y_pred):
        """Calculates the derivative of the LogLoss for each prediction
        
        Returns:
            np.array: Partial derivative of the LogLoss
        """
        return -np.sum(y_true - y_pred)


class CrossEntropy(LossFunction):
    """Class for implementing the CrossEntropy loss function. Inheirits
    loss and loss_partial_derivative from LossFunction"""
    @classmethod
    def loss(cls, y_true, y_pred):
        """Calculates the CrossEntropy for the true vs predicted classes
        
        Returns:
            np.array: CrossEntropy for each input/class
        """
        return -np.mean(y_true*np.log(y_pred))

    @classmethod
    def loss_partial_derivative(cls, y_true, y_pred):
        """Calculates the derivative of the CrossEntropy for each prediction
        
        Returns:
            np.array: Partial derivative of the CrossEntropy
        """
        m = y_true.shape[0]
        grad = SoftmaxActivation.activation(y_pred)
        grad[range(m), y_true] -= 1
        grad = grad / m
        return grad
