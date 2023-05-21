import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import losses
from keras.utils import plot_model
from keras import backend as K


class Modelling:
    def __init__(self, input_dim, output_dim, learning_rate, no_of_layers, width, batch_size, memory, discount):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.no_of_layers = no_of_layers
        self.width = width
        self.batch_size = int(batch_size)
        self.model = self.model_it(no_of_layers, width)
        self.gamma = discount
        self.num_state = input_dim
        self.num_action = output_dim
        self.Memory = memory

    def model_it(self, no_of_layers, width):
        input1 = keras.Input(shape=(self.input_dim,)) # For input
        model1 = keras.Sequential()
        model1.add(Dense(width, activation='relu'))
        for a in range(no_of_layers):
            model1.add(Dense(width, activation='relu'))
        model1.add(Dense(self.output_dim, activation='linear'))  # For output
        output = model1(input1)
        model1 = keras.Model(inputs=input1, outputs=output, name='model')
        model1.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self.learning_rate))
        return model1

    def predict_one(self, state):
        state = np.reshape(state, [1, self.input_dim])
        return self.model.predict(state)

    def predict_batch(self, states):
        states = np.asarray(states).astype(np.float32)
        return self.model.predict(states)

    def train_batch(self, states, q_sa):
        self.model.fit(states, q_sa, epochs=1, verbose=0)

    def save_model(self, path, episode):
        self.model.save(os.path.join(path, 'trained_model '+str(episode)+'.h5'))

    def loop(self):
        batch = self.Memory.get_samples(self.batch_size)
        if len(batch) > 0:
            current_state = []
            next_state = []
            for val in batch:
                current_state.append(val[0])
                next_state.append(val[3])
            current_q_list = self.predict_batch(current_state)
            future_q_list = self.predict_batch(next_state)

            x = np.zeros((len(batch), self.num_state))
            y = np.zeros((len(batch), self.num_action))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]
                current_q = current_q_list[i]
                current_q[action] = reward + self.gamma * np.max(future_q_list[i])
                x[i] = state
                y[i] = current_q
            self.train_batch(x, y)




