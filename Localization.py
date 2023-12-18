from sklearn.cluster import DBSCAN
from constants import a
import matplotlib.pyplot as plt
import numpy as np


def _alteredEuclideanDist(p1, p2, a):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + a * ((p1[2] - p2[2]) ** 2)


def DBscan(data):
    if data.shape[1] != 3:
        raise ValueError("Input data should have shape (n_samples, 3)")

    # Create a DBSCAN instance with a custom distance metric
    dbscan = DBSCAN(
        eps=0.5, min_samples=5, metric=_alteredEuclideanDist, metric_params={"a": a}
    )

    # Fit and predict
    labels = dbscan.fit_predict(data)

    return labels

    # # Plot the results in 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap="viridis")

    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.set_title("DBSCAN Clustering Results in 3D")

    # plt.show()
