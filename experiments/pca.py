import os

import numpy as np
from sklearn.decomposition import PCA

from environment import rooms


def run_pca_experiments():
    pca = PCA(n_components=148)
    layout_paths = os.listdir('../layouts')
    layout_paths = ["../layouts/" + layout_path for layout_path in layout_paths]
    x = [np.array(rooms.map_to_flattened_matrix(layout_path)) for layout_path in layout_paths]
    pca.fit(x)
    sum = np.sum(pca.explained_variance_ratio_)
    print(sum)


if __name__ == '__main__':
    run_pca_experiments()
