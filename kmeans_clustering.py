import os.path
import shutil

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from sklearn.cluster import KMeans

import rooms

image.LOAD_TRUNCATED_IMAGES = True
model = VGG16(weights='imagenet', include_top=False)

# Variables
imdir = "layouts/"
targetdir = "kmeans_clusteredLayouts_count_of_obstacles/"
number_clusters = 10

# Loop over files and get features
layout_paths = os.listdir('layouts')
layout_paths = ["layouts/" + layout_path for layout_path in layout_paths]
# first try
# featurelist = [rooms.map_to_flattened_matrix(layout_path) for layout_path in layout_paths]
# second try with the count of obstacles
featurelist = [rooms.count_of_obstacles(layout_path) for layout_path in layout_paths]

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
