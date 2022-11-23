import numpy as np
import random
import math
from gwu_nn.activation_functions import SigmoidActivation


def tanh_derivative(values):
    return 1.-values ** 2

def rand (a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b-a) + a

class LSTMParam:
    def __init__(self, mem_cell, x_dim):
        self.mem_cell = mem_cell
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell
    # Weights for matrices
        self.wg = rand(-0.1, 0.1, mem_cell, concat_len)
        self.wi = rand(-0.1, 0.1, mem_cell, concat_len)
        self.wf = rand(-0.1, 0.1, mem_cell, concat_len)
        self.wo = rand(-0.1, 0.1, mem_cell, concat_len)
    # Weights for biases
        self.bg = rand(-0.1, 0.1, mem_cell) 
        self.bi = rand(-0.1, 0.1, mem_cell)
        self.bf = rand(-0.1, 0.1, mem_cell)
        self.bo = rand(-0.1, 0.1, mem_cell) 
    # Derivative of loss function
        self.wg_diff = np.zeros((mem_cell, concat_len))
        self.wi_diff = np.zeros((mem_cell, concat_len))
        self.wf_diff = np.zeros((mem_cell, concat_len))
        self.wo_diff = np.zeros((mem_cell, concat_len))
        self.bg_diff = np.zeros(mem_cell)
        self.bi_diff = np.zeros(mem_cell)
        self.bf_diff = np.zeros(mem_cell)
        self.bo_diff = np.zeros(mem_cell)

    def apply_diff(self, lr=1):
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        # Set the diffs to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)

class LSTMState:
    def __init__(self, mem_cell, x_dim):
        self.g = np.zeros(mem_cell)
        self.i = np.zeros(mem_cell)
        self.f = np.zeros(mem_cell)
        self.o = np.zeros(mem_cell)
        self.s = np.zeros(mem_cell)
        self.h = np.zeros(mem_cell)
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)

class LSTMNode:
    def __init__(self, lstm_param, lstm_state):
        self.param = lstm_param
        self.state = lstm_state
        # non-recurrent input concatenated with recurrent input
        self.xc = None

    def bottom_data(self, x, s_prev=None, h_prev=None):
        # first node in network
        if s_prev is None: s_prev = np.zeros_like(self.state.s)
        if h_prev is None: h_prev = np.zeros_like(self.state.h)
        # save data to be used in back propagation
        self.s_prev = s_prev
        self.h_prev = h_prev
        # concat x(t) and h(t-1)
        xc = np.hstack((x, h_prev))
        # Sigmoid decides which values to take in, tanh transforms new tokens to vectors
        self.state.g = np.tanh(np.dot(self.param.wg, self.xc) + self.param.bg)
        self.state.i = SigmoidActivation().activation(np.dot(self.param.wi, self.xc) + self.param.bi)
        # Forget gate decides which values to forget
        self.state.f = SigmoidActivation().activation(np.dot(self.param.wf, self.xc) + self.param.bf)
        # Calculate output state
        self.state.o = SigmoidActivation().activation(np.dot(self.param.wo, self.xc) + self.param.bo)
        # calculate the present state
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        # Output h
        self.state.h = np.tanh(self.state.s) * self.state.o
        self.xc=xc