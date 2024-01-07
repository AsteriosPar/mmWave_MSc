from sklearn.cluster import DBSCAN
import constants as const
import matplotlib.pyplot as plt
from itertools import groupby
import numpy as np


def _altered_EuclideanDist(p1, p2):
    return (
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
        + const.DB_Z_WEIGHT * ((p1[2] - p2[2]) ** 2)
    )


def apply_DBscan(data):
    dbscan = DBSCAN(
        eps=const.DB_EPS,
        min_samples=const.DB_MIN_SAMPLES,
        metric=_altered_EuclideanDist,
    )

    # Clustering labels
    labels = dbscan.fit_predict(data)

    # label of -1 means noice so we exclude it
    unique_labels = set(labels) - {-1}

    # Create a dictionary to store points for each cluster
    clustered_points = {label: [] for label in unique_labels}

    # Iterate through data and assign points to clusters
    for i, label in enumerate(labels):
        if label != -1:
            clustered_points[label].append(data[i])

    # Convert the dictionary to a list of clustered points
    clusters = list(clustered_points.values())

    return clusters


def apply_constraints(detObj):
    data_coords = np.column_stack((detObj["x"], detObj["y"], detObj["z"]))
    # data_doppler = np.array(detObj["doppler"])
    # data_range = np.array(detObj["range"])

    ef_coords = np.empty((0, 3), dtype="int16")
    # ef_doppler = np.empty((0,), dtype="int16")

    # Parse every data point in the detObj
    for index in range(len(data_coords)):
        # Remove data points out of field of interest (range (and azimuth but not implemented yet))
        # Remove Static Clutter
        # if (
        #     data_range[index] > const.C_RANGE_MIN
        #     and data_range[index] < const.C_RANGE_MAX
        #     # and data_doppler[index] > const.C_DOPPLER_THRES
        # ):
        ef_coords = np.append(ef_coords, [data_coords[index, :]], axis=0)
        # ef_doppler = np.append(ef_doppler, data_doppler[index])

    # ef_coords has a shape of (M, 3)
    # return (ef_doppler, ef_coords)
    return ef_coords


# def split_clusters(points, labels):
#     sorted_data = sorted(zip(labels, points), key=lambda x: x[0])

#     # Use groupby to group points by label
#     grouped_points = {
#         label: [p[1] for p in group]
#         for label, group in groupby(sorted_data, key=lambda x: x[0])
#     }

#     # Convert the result to a list of lists (optional)
#     result = [grouped_points[label] for label in sorted(set(labels))]

#     return result
