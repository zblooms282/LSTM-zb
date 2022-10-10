Usage
======================================================================

.. _installation:

Installation
--------------

To use GWU_NN, first install it using pip:

.. code-block:: console

   (.venv) $ pip install gwu_nn

Creating Networks
--------------------

To create a placeholder for the network, create a network object
using ``gwu_nn.gwu_network.GWUNetwork()`` class:

.. autoclass:: gwu_nn.gwu_network.GWUNetwork

From the resulting object, it is possible to add additional layers to the network
using the ``GWUNetwork.add()`` method:

.. autoclass:: gwu_nn.layers.Layer
.. autofunction:: gwu_nn.gwu_network.GWUNetwork.add

Finally we can compile the model

For example:

>>> import numpy as np
>>> import pandas as pd
>>> from sklearn import preprocessing
>>> from sklearn.model_selection import train_test_split
>>>
>>> from gwu_nn.gwu_network import GWUNetwork
>>> from gwu_nn.layers import Dense
>>> from gwu_nn.activation_layers import Sigmoid
>>>
>>>
>>> y_col = 'Survived'
>>> x_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
>>> df = pd.read_csv('examples/data/titanic_data.csv')
>>> y = np.array(df[y_col]).reshape(-1, 1)
>>> orig_X = df[x_cols]
>>>
>>> # Lets standardize our features
>>> scaler = preprocessing.StandardScaler()
>>> stand_X = scaler.fit_transform(orig_X)
>>> X = stand_X
>>>
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
>>>
>>> network = GWUNetwork()
>>> network.add(Dense(14, add_bias=True, input_size=X.shape[1]))
>>> network.add(Dense(1, add_bias=True))
>>> network.add(Sigmoid())
>>> network.compile(loss='log_loss', lr=.01)
>>> network.fit(X_train, y_train, batch_size=10, epochs=100)
>>>
>>> predictions = network.predict(X_test)

