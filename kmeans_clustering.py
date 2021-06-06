from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
import numpy as np
import rooms
from sklearn.cluster import KMeans
import shutil, os.path
image.LOAD_TRUNCATED_IMAGES = True
model = VGG16(weights='imagenet', include_top=False)

# Variables
imdir = "layouts/"
targetdir = "kmeans_clusteredLayouts/"
number_clusters = 10

# Loop over files and get features
layout_paths = os.listdir('layouts')
layout_paths = ["layouts/" + layout_path for layout_path in layout_paths]
featurelist = [rooms.map_to_flattened_matrix(layout_path) for layout_path in layout_paths]

#max_list_len = max([len(i) for i in featurelist])

#for i in range(len(featurelist)):
#    featurelist[i].extend([[0,0]] * (max_list_len - len(featurelist[i])))

# Clustering
kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(featurelist))

# Copy images renamed by cluster
# Check if target dir exists
try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_):
    print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
    shutil.copy(layout_paths[i], targetdir + str(m) + "_" + str(i) + ".txt")