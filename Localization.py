from sklearn.cluster import DBSCAN
import constants as const
import math
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
    input_data = np.column_stack(
        (detObj["x"], detObj["y"], detObj["z"], detObj["doppler"])
    )

    ef_data = np.empty((0, const.MOTION_MODEL.EKF_DIM[1]), dtype="float")

    for index in range(len(input_data)):
        # if input_data[index][3] > const.C_DOPPLER_THRES:
        # Transform the radial velocity into Cartesian
        r = math.sqrt(
            input_data[index, 0] ** 2
            + input_data[index, 1] ** 2
            + input_data[index, 2] ** 2
        )
        vx = input_data[index, 3] * input_data[index, 0] / r
        vy = input_data[index, 3] * input_data[index, 1] / r
        vz = input_data[index, 3] * input_data[index, 2] / r

        ef_data = np.append(
            ef_data,
            [
                np.array(
                    [
                        input_data[index, 0],
                        input_data[index, 1],
                        input_data[index, 2],
                        vx,
                        vy,
                        vz,
                    ]
                )
            ],
            axis=0,
        )

    return ef_data
