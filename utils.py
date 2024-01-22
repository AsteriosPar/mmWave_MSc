from sklearn.cluster import DBSCAN
import constants as const
import math
import numpy as np


def altered_EuclideanDist(p1, p2):
    # NOTE: The z-axis has less weight in the distance metric since the sillouette of a person is tall and thin.
    # Also, the further away from the sensor the more sparse the points, so we need a weighing factor
    weight = 1 - ((p1[1] + p2[1]) / 2) * const.DB_RANGE_WEIGHT
    return weight * (
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
        + const.DB_Z_WEIGHT * ((p1[2] - p2[2]) ** 2)
    )


def apply_DBscan(pointcloud, eps=const.DB_EPS, min_samples=const.DB_MIN_SAMPLES):
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
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
    """
    Preprocesses the point cloud data from the sensor.

    This function filters the input point cloud, converts radial to Cartesian velocity,
    and transforms the coordinates to the standard vertical-horizontal plane axis system.

    - Performs a Doppler check and applies static clutter filtering if enabled.
    - Transforms radial velocity into Cartesian components.
    - Translates points to a new coordinate system.
    - Applies scene constraints filtering based on z-coordinate and y-coordinate.


    Parameters
    ----------
    detObj : dict
        Dictionary containing the raw detection data with keys:
        - "x": x-coordinate
        - "y": y-coordinate
        - "z": z-coordinate
        - "doppler": Doppler velocity

    Returns
    -------
    np.ndarray
        Preprocessed data in the standard vertical-horizontal plane axis system.
        Columns:
        - x-coordinate
        - y-coordinate
        - z-coordinate
        - Cartesian velocity along the x-axis
        - Cartesian velocity along the y-axis
        - Cartesian velocity along the z-axis

    """
    input_data = np.column_stack(
        (detObj["x"], detObj["y"], detObj["z"], detObj["doppler"])
    )

    ef_data = np.empty((0, const.MOTION_MODEL.KF_DIM[1]), dtype="float")

    for index in range(len(input_data)):
        # Performs doppler check
        if input_data[index][3] > 0 or not const.ENABLE_STATIC_CLUTTER:
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

            # Perform scene constraints filtering
            # TODO: Add more scene constraints
            if (
                transformed_point[2] <= 2
                and transformed_point[2] > -0.1
                and transformed_point[1] > 0
            ):
                ef_data = np.append(
                    ef_data,
                    [transformed_point],
                    axis=0,
                )

    return ef_data
