import os
import numpy as np
import sys
from tensorflow import keras
from keras import layers
from keras import losses
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # kill warning about tensorflow


class TrainingModel:
    def __init__(self, no_of_layers, batch_size, learning_rate, input_dim, output_dim, wid_each_layer):
        self._no_of_layers = no_of_layers
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._wid_each_layer = wid_each_layer
        self._model = self.model(no_of_layers, wid_each_layer)

    def model(self, num_of_layers, width):
        input1 = keras.Input(shape=self._input_dim)
        model1 = keras.Sequential()
        model1.add(layers.Dense(width, activation='relu'))  # For input
        for a in range(num_of_layers):
            model1.add(layers.Dense(width, activation='relu'))
        model1.add(layers.Dense(width, activation='linear'))  # For output

        output = model1(input1)
        model1.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
        return model1

    def predict_one(self, state):

        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)

    def predict_batch(self, states):
        return self._model.predict(states)

    def train_batch(self, states, q_sa):
        self._model.fit(states, q_sa, epochs=1, verbose=0)

    def save_model(self, path):
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True,
                   show_layer_names=True)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def batch_size(self):
        return self._batch_size
