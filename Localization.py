from sklearn.cluster import DBSCAN
import constants as const
import matplotlib.pyplot as plt
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
    return dbscan.fit_predict(data)


def apply_constraints(detObj):
    data_coords = np.column_stack((detObj["x"], detObj["y"], detObj["z"]))
    data_doppler = np.array(detObj["doppler"])
    data_range = np.array(detObj["range"])

    ef_coords = np.empty((0, 3), dtype="int16")
    ef_doppler = np.empty((0,), dtype="int16")

    # Parse every data point in the detObj
    for index in range(len(data_range)):
        # Remove data points out of field of interest (range (and azimuth but not implemented yet))
        # Remove Static Clutter
        if (
            data_range[index] > const.C_RANGE_MIN
            and data_range[index] < const.C_RANGE_MAX
            and data_doppler[index] > const.C_DOPPLER_THRES
        ):
            ef_coords = np.append(ef_coords, [data_coords[index, :]], axis=0)
            ef_doppler = np.append(ef_doppler, data_doppler[index])

    # ef_coords has a shape of (M, 3)
    return (ef_doppler, ef_coords)
