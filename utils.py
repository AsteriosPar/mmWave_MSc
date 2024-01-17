from sklearn.cluster import DBSCAN
import constants as const
import math
import numpy as np


def altered_EuclideanDist(p1, p2):
    # NOTE: The z-axis is let a bit looser since the sillouette of a person is tall and thin.
    # Also, the further away from the sensor the more sparse the points, so we need a weighing factor
    weight = 1 - ((p1[1] + p2[1]) / 2) * const.DB_RANGE_WEIGHT
    return weight * (
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
        + const.DB_Z_WEIGHT * ((p1[2] - p2[2]) ** 2)
    )


def apply_DBscan(pointcloud):
    dbscan = DBSCAN(
        eps=const.DB_EPS,
        min_samples=const.DB_MIN_SAMPLES,
        metric=altered_EuclideanDist,
    )

    labels = dbscan.fit_predict(pointcloud)

    # label of -1 means noice so we exclude it
    filtered_labels = set(labels) - {-1}

    # Assign points to clusters
    clustered_points = {label: [] for label in filtered_labels}
    for i, label in enumerate(labels):
        if label != -1:
            clustered_points[label].append(pointcloud[i])

    # Return a list of clustered pointclouds
    clusters = list(clustered_points.values())
    return clusters


def point_transform_to_standard_axis(input):
    # Translation Matrix (T)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, const.S_HEIGHT], [0, 0, 0, 1]])

    # Rotation Matrix (R_inv)
    ang_rad = np.radians(const.S_TILT)
    R_inv = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(ang_rad), -np.sin(ang_rad), 0],
            [0, np.sin(ang_rad), np.cos(ang_rad), 0],
            [0, 0, 0, 1],
        ]
    )

    coordinates = np.concatenate((input[:3], [1]))
    velocities = np.concatenate((input[3:], [0]))
    transformed_coords = np.dot(T, np.dot(R_inv, coordinates))
    transformed_velocities = np.dot(T, np.dot(R_inv, velocities))

    return np.array(
        [
            transformed_coords[0],
            transformed_coords[1],
            transformed_coords[2],
            transformed_velocities[0],
            transformed_velocities[1],
            transformed_velocities[2],
        ]
    )


def preprocess_data(detObj):
    """this function takes the pointcloud from the sensor, filters it, converts radial to cartesian velocity and transforms it to the standard vertical-horizontal plane axis system"""
    input_data = np.column_stack(
        (detObj["x"], detObj["y"], detObj["z"], detObj["doppler"])
    )

    ef_data = np.empty((0, const.MOTION_MODEL.KF_DIM[1]), dtype="float")

    for index in range(len(input_data)):
        # if input_data[index][3] > 0:
        # z-axis limits
        # designated area limits

        # Transform the radial velocity into Cartesian
        r = math.sqrt(
            input_data[index, 0] ** 2
            + input_data[index, 1] ** 2
            + input_data[index, 2] ** 2
        )
        vx = input_data[index, 3] * input_data[index, 0] / r
        vy = input_data[index, 3] * input_data[index, 1] / r
        vz = input_data[index, 3] * input_data[index, 2] / r

        # Translate points to new coordinate system
        transformed_point = point_transform_to_standard_axis(
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
        )

        ef_data = np.append(
            ef_data,
            [transformed_point],
            axis=0,
        )

    return ef_data
