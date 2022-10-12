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
        y_col = 'Survived'
        x_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        df = pd.read_csv('examples/data/titanic_data.csv')
        y = np.array(df[y_col]).reshape(-1, 1)
        orig_X = df[x_cols]

        # Lets standardize our features
        scaler = preprocessing.StandardScaler()
        stand_X = scaler.fit_transform(orig_X)
        X = stand_X

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        network = GWUNetwork()
        network.add(Dense(16, add_bias=False, activation='relu', input_size=X_train.shape[1]))
        network.add(Dense(1, add_bias=False, activation='sigmoid'))
        network.compile(loss='log_loss', lr=.01)
        network.fit(self.X_train, self.y_train, batch_size=10, epochs=100)

        self.network = network


    def test_regression_training(self):
        input_size = self.X_train.shape[1]
        network = GWUNetwork()
        network.add(Dense(16, add_bias=False, activation='relu', input_size=input_size))
        network.add(Dense(1, add_bias=False, activation='sigmoid'))
        network.compile(loss='log_loss', lr=.01)
        network.fit(self.X_train, self.y_train, batch_size=10, epochs=10)

    def test_deep_training(self):
        input_size = self.X_train.shape[1]
        network = GWUNetwork()
        network.add(Dense(16, add_bias=False, activation='relu', input_size=input_size))
        network.add(Dense(8, add_bias=False, activation='relu'))
        network.add(Dense(1, add_bias=False, activation='sigmoid'))
        network.compile(loss='log_loss', lr=.01)
        network.fit(self.X_train, self.y_train, batch_size=10, epochs=10)

    def test_regression_prediction(self):
        preds = self.network.predict(self.X_test[:2])
        self.assertEqual(preds.shape, (2, 1))

    def test_regression_evaluate(self):
        loss = self.network.evaluate(self.X_test, self.y_test)
        self.assertTrue(loss > 0 and loss < 1)
