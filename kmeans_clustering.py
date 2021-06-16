import os.path
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from sklearn.cluster import KMeans
from tensorflow.keras import layers, losses
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.decomposition import PCA

import rooms

image.LOAD_TRUNCATED_IMAGES = True
model = VGG16(weights='imagenet', include_top=False)

# Variables
imdir = "layouts/"
targetdir = "kmeans_clusteredLayouts_count_of_obstacles/"
number_clusters = 10

latent_dim = 64


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(15 * 15, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def runPCA(featurelist):
    pca = PCA(n_components = 148)
    pca.fit(featurelist)
    sum = np.sum(pca.explained_variance_ratio_)
    print(sum)

def trainAutoencoder(featurelist):
    dataset_size = 1001
    batch_size = 16
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    autoencoder = Autoencoder(latent_dim)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    x_train = np.array(featurelist)
    x_train = x_train.astype('float32') / 255.
    x_train = x_train[..., tf.newaxis]

    autoencoder.fit(x_train, x_train,
                    epochs=100,
                    batch_size=64,
                    shuffle=True,
                    validation_split=0.2)

    # print(tf.squeeze(x_train[0]))

    encoded_imgs = autoencoder.encoder(x_train).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    # encoded_imgs = autoencoder.encoder(featurelist[0]).numpy() # das is falsch
    # decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


def kMeans_clustering():
    # Loop over files and get features
    layout_paths = os.listdir('layouts')
    layout_paths = ["layouts/" + layout_path for layout_path in layout_paths]
    pathlib.Path('kmeans_clusteredLayouts_count_of_obstacles').mkdir(parents=True, exist_ok=True)
    # first try
    # featurelist = [rooms.map_to_flattened_matrix(layout_path) for layout_path in layout_paths]

    # second try with the count of obstacles
    # featurelist = [rooms.count_of_obstacles(layout_path) for layout_path in layout_paths]

    # with autodecoder
    featurelist = [np.array(rooms.map_to_flattened_matrix(layout_path)) for layout_path in layout_paths]
    trainAutoencoder(featurelist)

    #with PCA
    #featurelist = [np.array(rooms.map_to_flattened_matrix(layout_path)) for layout_path in layout_paths]
    #runPCA(featurelist)

    # max_list_len = max([len(i) for i in featurelist])

    # for i in range(len(featurelist)):
    #    featurelist[i].extend([[0,0]] * (max_list_len - len(featurelist[i])))

    # Clustering
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


kMeans_clustering()
