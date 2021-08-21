import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten
from tensorflow.python.keras.models import Model

from environment import rooms


class Autoencoder(Model):

    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()

        self.encoder = Sequential([
            Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = Sequential([
            Dense(12 * 12, activation='sigmoid')
        ])

    def call(self, inputs, **kwargs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        pass

    def train(self, training_data):
        self.compile(optimizer='adam', loss='binary_crossentropy')

        x_train = np.asarray(training_data).astype(np.float32)
        x_train = x_train.astype('float32') / 2.
        x_train = x_train[..., tf.newaxis]

        history = self.fit(x_train, x_train,
                           epochs=100,
                           batch_size=64,
                           shuffle=True,
                           validation_split=0.2)

        prabelli = np.array(training_data[0].reshape(12, 12))
        prabelli2 = np.array(training_data[0].reshape(12, 12))

        encoded_imgs = self.encoder(x_train).numpy()
        decoded_imgs = self.decoder(encoded_imgs).numpy()
        dekkelsmouk = decoded_imgs[0].reshape((12, 12)).astype('float32') * 2.
        dekkelsmouk2 = decoded_imgs[0].reshape((12, 12)).astype('float32') * 2.

        ax = plt.axes()
        ax.set_facecolor('white')
        plt.title('Autoencoder validation loss')
        plt.xlabel('Episode')
        plt.ylabel('Mean Squared Error')
        plt.ylim(-0.1, 1.1)

        plt.plot(history.history['val_loss'])

        plt.show()

        sns.heatmap(dekkelsmouk)
        plt.show()

        sns.heatmap(prabelli)
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        sns.heatmap(dekkelsmouk2, ax=ax1)
        sns.heatmap(prabelli2, ax=ax2)
        plt.show()

        return history


class ConvolutionalAutoencoder(Model):

    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()

        self.encoder = Sequential([
            Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same',
                   input_shape=(12, 12, 1)),
            MaxPooling2D(pool_size=(2, 2), padding='same'),

            Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), padding='same'),

            Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), padding='same')
        ])

        self.decoder = Sequential([
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),

            Conv2D(16, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),

            Conv2D(32, (3, 3), activation='relu'),
            UpSampling2D((2, 2)),
            Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, inputs, **kwargs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        pass

    def train(self, training_data, visualize_loss=True, show_examples=True):
        self.compile(optimizer='adam', loss='binary_crossentropy')

        x_train = np.array(training_data)
        x_train = x_train.astype('float32') / 2.
        x_train = x_train[..., tf.newaxis]

        history = self.fit(x_train, x_train,
                           epochs=1,
                           batch_size=64,
                           shuffle=True,
                           validation_split=0.2)

        if visualize_loss:
            ax = plt.axes()
            ax.set_facecolor('white')

            plt.xlabel('Episode')
            plt.ylabel('Mean Squared Error')
            plt.ylim(-0.1, 1.1)

            plt.plot(history.history['val_loss'])
            plt.show()

        if show_examples:
            prabelli = np.array(x_train[0].reshape(12, 12))

            encoded_imgs = self.encoder(x_train).numpy()
            decoded_imgs = self.decoder(encoded_imgs).numpy()
            dekkelsmouk = decoded_imgs[0].reshape((12, 12)).astype('float32') * 2.

            sns.heatmap(dekkelsmouk)
            plt.show()

            sns.heatmap(prabelli)
            plt.show()

        return history


if __name__ == '__main__':
    autoencoder = Autoencoder(81)

    layout_paths = os.listdir('../layouts')
    layout_paths = ["../layouts/" + layout_path for layout_path in layout_paths]

    x = [np.array(rooms.map_to_flattened_matrix(layout_path)) for layout_path in layout_paths]

    autoencoder.train(x)
