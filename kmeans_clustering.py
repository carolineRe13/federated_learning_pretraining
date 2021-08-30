import itertools
import os.path
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Model,  model_from_json
from keras.preprocessing import image
from matplotlib import cm
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, silhouette_samples, silhouette_score
from tensorflow.keras import layers, losses
from collections import Counter, defaultdict
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam

import pandas as pd
from yellowbrick.cluster import KElbowVisualizer

np.random.seed(0)
import seaborn as sns

sns.set_theme()

import rooms

image.LOAD_TRUNCATED_IMAGES = True
model = VGG16(weights='imagenet', include_top=False)

# Variables
imdir = "layouts/"
targetdir = "kmeans_clusteredLayouts_one_hot/"
number_clusters = 18

latent_dim = 8 * 8

LATENT_DIMS = [2 * 2, 3 * 3, 4 * 4, 5 * 5, 6 * 6, 7 * 7, 8 * 8, 9 * 9, 10 * 10]
LABELS = ['2 x 2', '3 x 3', '4 x 4', '5 x 5', '6 x 6', '7 x 7', '8 x 8', '9 x 9', '10 x 10']

LATENT_DIMS_ROOM_COMPARAISON = [6 * 6, 7 * 7, 8 * 8]
LABELS_ROOM_COMPARAISON = ['6 x 6', '7 x 7', '8 x 8']


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        # not hot
        # self.decoder = tf.keras.Sequential([
        #    layers.Dense(10 * 10, activation='sigmoid')
        # ])
        # hot
        self.decoder = tf.keras.Sequential([
            layers.Dense(30 * 10, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def runPCA(featurelist):
    pca = PCA(n_components='mle')
    print(pca.n_components)
    pca.fit(featurelist)
    sum = np.sum(pca.explained_variance_ratio_)
    print(sum)
    print(len(pca.explained_variance_))
    np.set_printoptions(precision=3)
    print(pca.explained_variance_ratio_)
    singular_values = pca.explained_variance_ratio_
    # print(pca.explained_variance_)
    current_sum = 0
    results = []
    singular_values = np.cumsum(singular_values)
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance')
    plt.plot(singular_values)
    plt.show()



def evaluation_line_plot(results):
    marker = itertools.cycle(('o', '^', 's', 'd'))
    linestyle = itertools.cycle((':', '-.', '-'))

    plt.xlabel('Episode')
    plt.ylabel('Binary Crossentropy')
    plt.ylim(0, 1)
    #ax = plt.axes()
    #ax.set_facecolor("white")

    for scores, label in zip(results, LABELS):
        #plt.plot(scores, label=label, linestyle=next(linestyle), marker=next(marker))
        plt.plot(scores, label=label, linestyle=next(linestyle))

    plt.legend()
    plt.show()


def train_autoencoder(featurelist, latent_dim):
    dataset_size = 10001
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
    # sdg
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.CosineSimilarity(axis=1))
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    x_train = np.array(featurelist)
    x_train = x_train.astype('float32') / 20.
    x_train = x_train[..., tf.newaxis]
    history = autoencoder.fit(x_train, x_train,
                    epochs=100,
                    batch_size=64,
                    shuffle=True,
                    validation_split=0.2)

    # not hot
    # prabelli = np.array(featurelist[0].reshape(10, 10))
    # prabelli2 = np.array(featurelist[1].reshape(10, 10))

    # hot
    # prabelli = np.array(reshape_hot(featurelist[0])).reshape(10, 10)
    encoded_imgs = autoencoder.encoder(x_train).numpy()
    # decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    # dekkelsmouk = np.array(reshape_hot_arg_max(decoded_imgs[0])).reshape((10, 10))

    # not hot
    # encoded_imgs = autoencoder.encoder(x_train).numpy()
    # decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    # dekkelsmouk = decoded_imgs[0].reshape((10, 10)).astype('float32') * 2.
    #dekkelsmouk2 = decoded_imgs[1].reshape((10, 10)).astype('float32') * 2.

    return encoded_imgs
    # return history.history['val_loss']

    # plt.title('Autoencoder validation loss')
    # plt.xlabel('Episode')
    # plt.ylabel('Cosine Embedding Loss')
    # plt.ylim(-1, 1)

    # plt.plot(history.history['val_loss'])

    # plt.show()

    # print(history.history['val_loss'])

    # total_error = np.sum(np.absolute(prabelli - dekkelsmouk))

    # Save the weights
    autoencoder.save_weights('./checkpoints/my_checkpoint')

    # Create a new model instance
    autoencoder = Autoencoder(latent_dim)

    # Restore the weights
    autoencoder.load_weights('./checkpoints/my_checkpoint')

    #print(total_error)

    #print(mean_absolute_error(prabelli, dekkelsmouk))

    #fig, (ax1, ax2) = plt.subplots(1, 2)
    # sns.heatmap(prabelli2, cbar=False, xticklabels=False, yticklabels=False)
    #plt.show()
    # sns.heatmap(prabelli, cbar=False, xticklabels=False, yticklabels=False)
    #plt.show()

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # sns.heatmap(dekkelsmouk2, ax=ax1)
    # sns.heatmap(prabelli2, ax=ax2)
    # plt.show()

    # return history.history['val_loss']

    # return encoded_imgs


def reshape_hot(room):
    reshaped_room = []
    for x in range(0, 298, 3):
        field = []
        for y in range(0, 3):
            field.append(room[x + y])
        if field == [0, 0, 1]:
            reshaped_room.append(0)
        elif field == [0, 1, 0]:
            reshaped_room.append(1)
        elif field == [1, 0, 0]:
            reshaped_room.append(2)
    return reshaped_room


def reshape_hot_arg_max(room):
    reshaped_room = []
    for x in range(0, 298, 3):
        field = []
        for y in range(0, 3):
            field.append(room[x + y])
        reshaped_room.append(np.argmax(field))
    return reshaped_room



def run_kmeans(featurelist, layout_paths):
    featurelist = np.array(featurelist)[:800]
    kmeans = KMeans(n_clusters = number_clusters).fit(featurelist)
    print(Counter(kmeans.labels_))
    try:
        os.makedirs(targetdir)
    except OSError:
        pass
    # Copy with cluster name
    print("\n")
    for i, m in enumerate(kmeans.labels_):
        print("    Copy: %s / %s" % (i, len(kmeans.labels_)), end="\r")
        shutil.copy(layout_paths[i], targetdir + str(m) + "_" + str(i) + ".txt")

def run_silhouette(featurelist, layout_paths):

    featurelist = np.array(featurelist)[:800]
    scores = []
    for k in range(2, 30):
        # kmeans
        # kmeans = KMeans(n_clusters=k).fit(featurelist)
        agc = AgglomerativeClustering(distance_threshold=None,
                                      n_clusters=k,
                                      affinity="manhattan",
                                      linkage="complete",
                                      )
        agc = agc.fit(featurelist)
        score = silhouette_score(featurelist, agc.labels_)
        scores.append(score)
        print(k, score)

    plt.xlabel('number of clusters')
    plt.ylabel('silhouette score')
    plt.ylim(-1, 1)

    plt.plot(range(3, 31), scores)
    plt.show()


def run_spectral(featurelist, layout_paths):
    featurelist = np.array(featurelist)[:800]
    scores = []
    # kmeans
    # kmeans = KMeans(n_clusters=k).fit(featurelist)
    spectral = SpectralClustering(n_clusters=18)

    spectral.fit(featurelist)

    try:
        os.makedirs("spectral_clustered_layouts/")
    except OSError:
        pass
    # Copy with cluster name
    print("\n")
    for i, m in enumerate(spectral.labels_):
        print("    Copy: %s / %s" % (i, len(spectral.labels_)), end="\r")
        shutil.copy(layout_paths[i], "spectral_clustered_layouts/" + str(m) + "_" + str(i) + ".txt")


def autoencoder_kmeans():
    layout_paths = os.listdir('layouts')
    layout_paths = ["layouts/" + layout_path for layout_path in layout_paths]
    layout_paths_flattened = [np.array(rooms.map_to_flattened_one_hot_matrix(layout_path)) for layout_path in layout_paths]
    run_kmeans(train_autoencoder(layout_paths_flattened, 7*7), layout_paths)


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
    # featurelist = [np.array(rooms.map_to_flattened_matrix(layout_path)) for layout_path in layout_paths]
    # train_autoencoder(featurelist, latent_dim)

    # autoencoder with one-hot
    featurelist = [np.array(rooms.map_to_flattened_one_hot_matrix(layout_path)) for layout_path in layout_paths]
    # train_autoencoder(featurelist, latent_dim)

    # classic
    run_kmeans(train_autoencoder(featurelist, 7 * 7), layout_paths)

    # spectral
    # run_spectral(train_autoencoder(featurelist, 7 * 7), layout_paths)

    # elbow(train_autoencoder(featurelist, 7 * 7))

    # gap_statistic(train_autoencoder(featurelist, 7 * 7))

    # optimalK(train_autoencoder(featurelist, 7 * 7))

    #calculate_WSS(train_autoencoder(featurelist, 7 * 7),1000)

    # with PCA
    # featurelist = [np.array(rooms.map_to_flattened_matrix(layout_path)) for layout_path in layout_paths]
    # runPCA(featurelist)

    # max_list_len = max([len(i) for i in featurelist])

    # for i in range(len(featurelist)):
    #    featurelist[i].extend([[0,0]] * (max_list_len - len(featurelist[i])))

    # Clustering
    # kmeans = KMeans(n_clusters=number_clusters).fit(np.array(featurelist))


    # Copy images renamed by cluster
    # Check if target dir exists
    try:
        os.makedirs(targetdir)
    except OSError:
        pass
    # Copy with cluster name
    print("\n")
    # for i, m in enumerate(kmeans.labels_):
    #    print("    Copy: %s / %s" % (i, len(kmeans.labels_)), end="\r")
    #    shutil.copy(layout_paths[i], targetdir + str(m) + "_" + str(i) + ".txt")


def val_loss_comparaison(): # geiles englisch haste hier am start :P xDDD
    layout_paths = os.listdir('layouts')
    layout_paths = ["layouts/" + layout_path for layout_path in layout_paths]
    pathlib.Path('kmeans_clusteredLayouts_count_of_obstacles').mkdir(parents=True, exist_ok=True)
    # featurelist = [np.array(rooms.map_to_flattened_matrix(layout_path)) for layout_path in layout_paths]
    featurelist = [np.array(rooms.map_to_flattened_matrix(layout_path)) for layout_path in
                   layout_paths]

    results = []

    for latent_dim in LATENT_DIMS:
        results.append(train_autoencoder(featurelist, latent_dim))

    evaluation_line_plot(results)


def latent_dim_comparaison():
    layout_paths = os.listdir('layouts')
    layout_paths = ["layouts/" + layout_path for layout_path in layout_paths]
    pathlib.Path('kmeans_clusteredLayouts_count_of_obstacles').mkdir(parents=True, exist_ok=True)
    featurelist = [np.array(rooms.map_to_flattened_one_hot_matrix(layout_path)) for layout_path in
                   layout_paths]

    dataset_size = 10001
    batch_size = 16
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42

    results = []

    # Creating data indices for training and validation splits:
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    for latent_dim in LATENT_DIMS_ROOM_COMPARAISON:
        autoencoder = Autoencoder(latent_dim)
        # autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
        autoencoder.compile(optimizer='adam', loss=tf.keras.losses.CosineSimilarity(axis=1))

        x_train = np.array(featurelist)
        x_train = x_train.astype('float32') / 20.
        x_train = x_train[..., tf.newaxis]
        history = autoencoder.fit(x_train, x_train,
                              epochs=100,
                              batch_size=64,
                              shuffle=True,
                              validation_split=0.2)
        encoded_imgs = autoencoder.encoder(x_train).numpy()
        decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
        reshaped_decoded_img = decoded_imgs[0].reshape((10, 10)).astype('float32') * 2.
        results.append(reshaped_decoded_img)

    original_img = np.array(featurelist[0].reshape(10, 10)).astype('float32') / 20.

    # total_error = np.sum(np.absolute(original_img - reshaped_decoded_img))

    # Save the weights
    autoencoder.save_weights('./checkpoints/my_checkpoint')

    # Create a new model instance
    autoencoder = Autoencoder(latent_dim)

    # Restore the weights
    autoencoder.load_weights('./checkpoints/my_checkpoint')

    # print(total_error)

    # print(mean_absolute_error(prabelli, dekkelsmouk))

    sns.heatmap(original_img, xticklabels=False, yticklabels=False, vmin=0, vmax=1)
    plt.show()
    sns.heatmap(results[0], xticklabels=False, yticklabels=False, vmin=0, vmax=1)
    plt.show()
    sns.heatmap(results[1], xticklabels=False, yticklabels=False, vmin=0, vmax=1)
    plt.show()
    sns.heatmap(results[2], xticklabels=False, yticklabels=False, vmin=0, vmax=1)
    plt.show()


def val_loss_comparaison_cnn():
    layout_paths = os.listdir('layouts')
    layout_paths = ["layouts/" + layout_path for layout_path in layout_paths]
    pathlib.Path('kmeans_clusteredLayouts_count_of_obstacles').mkdir(parents=True, exist_ok=True)
    # featurelist = [np.array(rooms.map_to_flattened_matrix(layout_path)) for layout_path in layout_paths]
    featurelist = [np.array(rooms.map_to_flattened_matrix(layout_path)) for layout_path in
                   layout_paths]

    results = []

    for latent_dim in range(2,11):
        results.append(train_CNN_autoencoder(featurelist, latent_dim))

    evaluation_line_plot(results)



def train_CNN_autoencoder(featurelist):
    autoencoder = convolutional_model()
    autoencoder.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')
    autoencoder.summary()

    x_train = np.array(featurelist)
    x_train = x_train.astype('float32') / 2.
    x_train = x_train[..., tf.newaxis]
    history = autoencoder.fit(x_train, x_train,
                              epochs=100,
                              batch_size=64,
                              shuffle=True,
                              validation_split=0.2)

    # return history.history['val_loss']
    # print(autoencoder.predict(x_train))
    # encoded_imgs = autoencoder.encoder(x_train).numpy()
    # decoded_imgs = autoencoder.predict(x_train)[0]
    # original_image = decoded_imgs.reshape((10, 10)).astype('float32') * 2.
    # second_original_image = decoded_imgs[0].reshape((12, 12)).astype('float32') * 2.
    # original_image = decoded_imgs[0].reshape((12, 12)).astype('float32') * 2.
    marker = itertools.cycle(('o', '^', 's', 'd'))
    linestyle = itertools.cycle((':', '-.', '-'))

    plt.title('Autoencoder validation loss')
    plt.xlabel('Episode')
    plt.ylabel('Binary Cross-entrophy')
    plt.ylim(0, 1)

    plt.plot(history.history['val_loss'], linestyle=next(linestyle))

    plt.show()


def convolutional_model():
    # The encoding process
    input_img = Input(shape=(10, 10, 1))

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


def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        # kmeans
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    plt.xlabel('number of clusters')
    plt.ylabel('Within-Cluster-Sum of Squared Errors')
    plt.plot(sse)
    plt.show()


def elbow(featurelist):
    # model = KMeans()
    model = SpectralClustering()
    # k is range of number of clusters.
    # visualizer = KElbowVisualizer(model, k=(2, 50), metric='calinski_harabasz', timings=False, locate_elbow=False, title=' ')
    # visualizer = KElbowVisualizer(model, k=(2, 50), timings=False, locate_elbow=False, title=' ')
    visualizer = KElbowVisualizer(model, k=(2, 50), metric='silhouette', timings=False, locate_elbow=False, title=' ')
    visualizer.fit(featurelist)  # Fit data to visualizer
    plt.ylim(-1, 1)
    visualizer.show()


def kmeans(num_centers, X):
    centers = np.random.uniform(np.min(X[0]), np.max(X[0]), size=(num_centers, 2))

    for _ in range(10):

        assigned_to_clusters = np.zeros(len(X))

        for i, x in enumerate(X):
            closest_center = np.argmin([np.linalg.norm(x - center) for center in centers])
            assigned_to_clusters[i] = closest_center

        new_centers = np.array([np.mean(X[assigned_to_clusters == i], axis=0) for i in range(num_centers)])
        convergence = np.sum(np.abs(new_centers - centers))
        centers = new_centers

    return np.array(new_centers), np.array(assigned_to_clusters)


def calculate_Wk(Ks, input_data):
    e = np.finfo(float).eps
    if (len(input_data) == 2):
        X = input_data[0]
        label = input_data[1]
    else:
        X = input_data

    Ks = np.arange(1, Ks + 1)
    within_cluster_dists = []

    for K in Ks:

        dists = 0
        centroids, points = kmeans(K, X)

        for i in range(K):
            cluster_array = X[points == i]
            centroid_dist = []
            dist = 0

            if len(cluster_array) > 0:
                for j in range(len(cluster_array)):
                    centroid_dist.append(np.linalg.norm(centroids[i] - cluster_array[j]))

            dist += np.sum(centroid_dist)
            dists += dist + e

        within_cluster_dists.append(np.log(((dists + e) / K)))

        normalized_wcd = within_cluster_dists - np.max(within_cluster_dists)

    return normalized_wcd


def simulate(iterations, K):
    e = np.finfo(float).eps
    simulated_Wk = np.zeros((iterations, K)) + e
    simulated_sk = np.zeros((iterations, K)) + e

    for i in range(iterations):
        temp_wk = []
        temp_sk = []

        X = np.random.uniform(0, 1, size=(300, 2))

        within_cluster_dists = calculate_Wk(K, X)

        simulated_Wk[i] = within_cluster_dists

    Wks = np.mean(simulated_Wk + e, axis=0)
    sks = np.std(simulated_Wk + e, axis=0) * np.sqrt(1 + 1 / 20)

    return Wks, sks


def shift_to_one(shift, to_range):
    shifted = np.zeros(to_range)
    shifted[1:] = shift
    shifted[0] = np.nan
    return shifted

def gap(simulations, Ks, input_data):
    optimum = 0
    if (len(input_data) == 2):
        data = input_data[0]
        label = input_data[1]
    else:
        data = input_data
    Wks, sks = simulate(simulations, Ks)
    within_cluster_dists = calculate_Wk(Ks, data)
    G = Wks - within_cluster_dists
    shifted_g = shift_to_one(G, Ks + 1)

    # THE GAP STATISTICS FORMULA FROM THE SCRIPT
    for i in range(0, len(G) - 1):
        if (G[i] >= G[i + 1] + sks[i + 1]):
            optimum = i
            break

    if optimum == 0:
        optimum = 1

    plt.figure(figsize=(16, 10))

    # mapped_colors = [colors[i] for i in label]

    plt.subplot(2, 2, 3)
    plt.plot(shifted_g, '-o', color='r')
    plt.axvline(x=optimum, color='black')
    plt.title("Gap Statistics with optimum K at {}".format(optimum))


def optimalK(data, nrefs=3, maxClusters=50):
    """
    Calculates KMeans optimal K using Gap Statistic
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):
        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = SpectralClustering(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp
        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_
        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    score_g = gaps.argmax() + 1
    plt.plot(resultsdf['clusterCount'], resultsdf['gap'], linestyle='-', marker="D", color='#1f77b4')
    plt.xlabel('k')
    plt.ylabel('gap statistic')
    plt.show()


def within_sum_of_squares(data, centroids, labels):
    """
    Compute total sum of squares of a prototype clustering
    algorithm.
    """

    SSW = 0
    for l in np.unique(labels):
        data_l = data[labels == l]
        resid = data_l - centroids[l]
        SSW += (resid**2).sum()
    return SSW


def gap_statistic(featureList):
    gaps = np.zeros((len(range(1, 50)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, 50)):

        # Holder for reference dispersion results
        refDisps = np.zeros(3)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(3):
            # Create new random reference set
            randomReference = np.random.random_sample(size=featureList.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            # km = AgglomerativeClustering(distance_threshold=None,
            #                                n_clusters=k,
            #                                affinity="manhattan",
            #                                linkage="complete",
            #                                )

            #km = SpectralClustering(n_clusters=k)
            #km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        # km = model = AgglomerativeClustering(distance_threshold=None,
        #                              n_clusters=k,
        #                              affinity="manhattan",
        #                              linkage="complete",
        #                              )
        km.fit(featureList)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)
    plt.plot(resultsdf.clusterCount, resultsdf.gap, linewidth=3)
    plt.scatter(resultsdf[resultsdf.clusterCount == gaps.argmax() + 1].clusterCount, resultsdf[resultsdf.clusterCount == gaps.argmax() + 1].gap, s=250, c='r')
    plt.grid(True)
    plt.xlabel('Cluster Count')
    plt.ylabel('Gap Value')
    plt.show()
    print(resultsdf[resultsdf.clusterCount == gaps.argmax() + 1].clusterCount, resultsdf[resultsdf.clusterCount == gaps.argmax() + 1].gap)


if __name__ == '__main__':
    layout_paths = os.listdir('layouts')
    layout_paths = ["layouts/" + layout_path for layout_path in layout_paths]
    layout_paths_flattened = [np.array(rooms.map_to_flattened_matrix(layout_path)) for layout_path in
                              layout_paths]
    # train_autoencoder(layout_paths_flattened, 7*7)
    kMeans_clustering()
    # latent_dim_comparaison()
    # autoencoder_kmeans()

