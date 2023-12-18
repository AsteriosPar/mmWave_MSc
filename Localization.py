from sklearn.cluster import DBSCAN
from constants import a
import matplotlib.pyplot as plt
import numpy as np


def _altered_EuclideanDist(p1, p2, a):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + a * ((p1[2] - p2[2]) ** 2)


def apply_DBscan(data):
    dbscan = DBSCAN(
        eps=0.5, min_samples=5, metric=_altered_EuclideanDist, metric_params={"a": a}
    )

    # Fit and predict
    labels = dbscan.fit_predict(data)

    return labels


def apply_constraints(detObj):
    data_coords = np.column_stack((detObj["x"], detObj["y"], detObj["z"]))
    data_doppler = np.array(detObj["doppler"])
    data_range = np.array(detObj["range"])

    ef_coords = np.empty((0, 3), dtype="int16")
    ef_doppler = np.empty((0,), dtype="int16")

    # Parse every data point in the detObj
    for index in range(data_range.len()):
        # Remove data points out of field of interest (range (and azimuth but not implemented yet))
        # Remove Static Clutter

        if (
            data_range[index] > 0.1
            and data_range[index] < 15
            and data_doppler[index] > 0
        ):
            ef_coords = np.append(ef_coords, data_coords[index, :], axis=0)
            ef_doppler = np.append(ef_doppler, data_doppler[index])

    # ef_coords has a shape of (M, 3)
    return (ef_doppler, ef_coords)
