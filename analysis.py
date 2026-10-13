import sys

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

import symnmfmodule
from kmeans import kmeans
from symnmf import initial_H, read_points


def symnmf_labels(points, k):
    """Cluster points with SymNMF and return the resulting labels."""
    data = points.tolist()
    w = symnmfmodule.norm(data)
    h = symnmfmodule.symnmf(initial_H(w, k), w)
    return list(np.argmax(h, axis=1))


def main():
    """Cluster the dataset both ways and print the three comparison scores."""
    # k and file_name may be assumed legal, so no validation here.
    k = int(sys.argv[1])
    points = read_points(sys.argv[2])
    nmf_clusters = symnmf_labels(points, k)
    kmeans_clusters = kmeans(points.tolist(), k)
    print("nmf: %.4f" % silhouette_score(points, nmf_clusters))
    print("kmeans: %.4f" % silhouette_score(points, kmeans_clusters))
    print("ari: %.4f" % adjusted_rand_score(nmf_clusters, kmeans_clusters))


if __name__ == "__main__":
    main()
