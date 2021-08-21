import os.path
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from sklearn.decomposition import PCA
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.clustering_ops import KMeans

sns.set_theme()

from environment import rooms

image.LOAD_TRUNCATED_IMAGES = True
model = VGG16(weights='imagenet', include_top=False)

# Variables
imdir = "layouts/"
targetdir = "kmeans_clusteredLayouts_count_of_obstacles/"
number_clusters = 10

latent_dim = 10 * 10


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(12 * 12, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def run_PCA(featurelist):
    pca = PCA(n_components=144)
    pca.fit(featurelist)
    sum = np.sum(pca.explained_variance_ratio_)
    print('PCA:', pca.va)
    print('PCA sum:', sum)


def train_autoencoder(featurelist):
    dataset_size = 10001
    batch_size = 16
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    # indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * dataset_size))
    # if shuffle_dataset:
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]

    autoencoder = Autoencoder(latent_dim)
    autoencoder.compile(optimizer='adam', loss='mse')

    x_train = np.array(featurelist)
    x_train = x_train.astype('float32') / 2.
    x_train = x_train[..., tf.newaxis]
    history = autoencoder.fit(x_train, x_train,
                              epochs=1000,
                              batch_size=64,
                              shuffle=True,
                              validation_split=0.2)

    first_decoded_image = np.array(featurelist[0].reshape(12, 12))
    second_decoded_image = np.array(featurelist[0].reshape(12, 12))

    encoded_imgs = autoencoder.encoder(x_train).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    original_image = decoded_imgs[0].reshape((12, 12)).astype('float32') * 2.
    second_original_image = decoded_imgs[0].reshape((12, 12)).astype('float32') * 2.

    ax = plt.axes()
    ax.set_facecolor('white')
    plt.title('Autoencoder validation loss')
    plt.xlabel('Episode')
    plt.ylabel('Mean Squared Error')
    plt.ylim(-0.1, 1.1)

    plt.plot(history.history['val_loss'])

    plt.show()

    # print(history.history['val_loss'])

    total_error = np.sum(np.absolute(first_decoded_image - original_image))

    # Save the weights
    autoencoder.save_weights('./checkpoints/my_checkpoint', overwrite=True)

    # Create a new model instance
    autoencoder = Autoencoder(latent_dim)

    # Restore the weights
    autoencoder.load_weights('./checkpoints/my_checkpoint')

    # print(total_error)

    # print(mean_absolute_error(first_decoded_image, original_image))

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # sns.heatmap(original_image, ax=ax1)
    # sns.heatmap(first_decoded_image, ax=ax2)
    # plt.show()
    sns.heatmap(original_image)
    plt.show()

    sns.heatmap(first_decoded_image)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.heatmap(second_original_image, ax=ax1)
    sns.heatmap(second_decoded_image, ax=ax2)
    plt.show()


def run_autoencoder(featurelist, layout_paths):
    model = Autoencoder(latent_dim)
    # Loads the weights
    # model.load_weights('./checkpoints/my_checkpoint')
    x_train = np.array(featurelist)
    x_train = x_train.astype('float32') / 2.
    x_train = x_train[..., tf.newaxis]
    predicted_results = []
    for layout in enumerate(x_train):
        predicted_results.append(
            model.predict((x_train)[layout]).reshape((12, 12)).astype('float32') * 2)
    layout_paths = np.array(layout_paths)
    run_kMeans(featurelist, predicted_results)


def train_CNN_autoencoder(featurelist):
    autoencoder = c()
    autoencoder.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')
    autoencoder.summary()

    x_train = np.array(featurelist)
    x_train = x_train.astype('float32') / 2.
    x_train = x_train[..., tf.newaxis]
    history = autoencoder.fit(x_train, x_train,
                              epochs=1000,
                              batch_size=64,
                              shuffle=True,
                              validation_split=0.2)

    # print(autoencoder.predict(x_train))
    # encoded_imgs = autoencoder.encoder(x_train).numpy()
    decoded_imgs = autoencoder.predict(x_train)[0]
    original_image = decoded_imgs.reshape((12, 12)).astype('float32') * 2.
    # second_original_image = decoded_imgs[0].reshape((12, 12)).astype('float32') * 2.
    # original_image = decoded_imgs[0].reshape((12, 12)).astype('float32') * 2.
    first_decoded_image = featurelist[0]
    plt.xticks([])
    plt.yticks([])
    ax = sns.heatmap(first_decoded_image)
    ax.axis('off')
    plt.show()
    ax = sns.heatmap(original_image)
    ax.axis('off')
    plt.show()


def c():
    # The encoding process
    input_img = Input(shape=(12, 12, 1))

    ############
    # Encoding #
    ############

    # Conv1 #
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Conv2 #
    x = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Conv 3 #
    x = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    ############
    # Decoding #
    ############

    # DeConv1
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)

    # DeConv2
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Deconv3
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    return Model(input_img, decoded)


def run_kMeans(featurelist, layout_paths):
    kmeans = KMeans(n_clusters=number_clusters).fit(np.array(featurelist))

    # Copy images renamed by cluster
    # Check if target dir exists
    try:
        os.makedirs(targetdir)
    except OSError:
        pass
    # Copy with cluster name
    print("\n")
    for i, m in enumerate(kmeans.labels_):
        print("    Copy: %s / %s" % (i, len(kmeans.labels_)), end="\r")
        shutil.copy(layout_paths[i], targetdir + str(m) + "_" + str(i) + ".txt")


def clustering():
    # Loop over files and get features
    layout_paths = os.listdir('layouts')
    layout_paths = ["layouts/" + layout_path for layout_path in layout_paths]
    pathlib.Path('kmeans_clusteredLayouts_count_of_obstacles').mkdir(parents=True, exist_ok=True)
    # first try
    # featurelist = [rooms.map_to_flattened_matrix(layout_path) for layout_path in layout_paths]

    # second try with the count of obstacles
    # featurelist = [rooms.count_of_obstacles(layout_path) for layout_path in layout_paths]

    # with autodecoder
    featurelist = [np.array(rooms.map_to_flattened_matrix(layout_path)) for layout_path in
                   layout_paths]
    # train_autoencoder(featurelist)
    run_autoencoder(featurelist, layout_paths)

    # CNN + autoencoder
    # featurelist = [rooms.map_to_matrix(layout_path) for layout_path in layout_paths]
    # train_CNN_autoencoder(featurelist)

    # with PCA
    # featurelist = [np.array(rooms.map_to_flattened_matrix(layout_path)) for layout_path in layout_paths]
    # run_PCA(featurelist)

    # max_list_len = max([len(i) for i in featurelist])

    # for i in range(len(featurelist)):
    #    featurelist[i].extend([[0,0]] * (max_list_len - len(featurelist[i])))


clustering()
