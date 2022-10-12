import unittest

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense
from gwu_nn.activation_layers import Sigmoid


class TestLogisticRegression(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(10)
        x = np.random.randint(-20, 20, size=(1000,2))
        y = np.prod(x, axis=1)

        # Lets standardize our features
        scaler = preprocessing.StandardScaler()
        stand_X = scaler.fit_transform(x)

        X_train, X_test, y_train, y_test = train_test_split(stand_X, y, test_size=0.33, random_state=42)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        network = GWUNetwork()
        network.add(Dense(8, add_bias=False, activation='relu', input_size=X_train.shape[1]))
        network.add(Dense(32, add_bias=False, activation='relu'))
        network.add(Dense(8, add_bias=False, activation='relu'))
        network.add(Dense(1, add_bias=False))
        network.compile(loss='mse', lr=.0001)
        network.fit(self.X_train, self.y_train, batch_size=25, epochs=100)

        self.network = network

    def test_regression_prediction(self):
        preds = self.network.predict(self.X_test[:2])
        self.assertEqual(preds.shape, (2, 1))

    def test_regression_evaluate(self):
        loss = self.network.evaluate(self.X_test, self.y_test)
        self.assertTrue(loss > 0 and loss < 5000)

    def test_regression_prediction_eval(self):
        ix = 157
        actual = self.y_test[ix]
        preds = self.network.predict(self.X_test[ix:ix+2])
        self.assertTrue(preds[0] > actual-5 and preds[0] < actual+5)