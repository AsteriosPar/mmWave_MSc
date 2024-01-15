from sklearn.cluster import DBSCAN
import constants as const
import math
import numpy as np
from Tracking import TrackBuffer


class BatchedData:
    def __init__(self):
        self.counter = 0
        self.effective_data = np.empty((0, const.MOTION_MODEL.KF_DIM[1]), dtype="float")

    def empty(self):
        self.counter = 0
        self.effective_data = np.empty((0, const.MOTION_MODEL.KF_DIM[1]), dtype="float")


def _altered_EuclideanDist(p1, p2):
    # NOTE: The z-axis is let a bit looser since the sillouette of a person is tall and thin.
    # Also, the further away from the sensor the more sparse the points, so we need a weighing factor
    weight = 1 - ((p1[1] + p2[1]) / 2) * const.DB_RANGE_WEIGHT
    return weight * (
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


def transform(input):
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

    x_transformed, y_transformed, z_transformed, _ = transformed_coords
    vx_transformed, vy_transformed, vz_transformed, _ = transformed_velocities

    return np.array(
        [
            x_transformed,
            y_transformed,
            z_transformed,
            vx_transformed,
            vy_transformed,
            vz_transformed,
        ]
    )


def apply_constraints(detObj):
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
        transformed_point = transform(
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


def batch_frames(batch: BatchedData, new_data: np.array):
    is_ready = False
    batch.effective_data = np.append(batch.effective_data, new_data, axis=0)

    if batch.counter < (const.FB_FRAMES_BATCH - 1):
        batch.counter += 1
    else:
        batch.counter = 0
        is_ready = True

    return is_ready


def perform_tracking(pointcloud, trackbuffer: TrackBuffer, batch: BatchedData):
    # Prediction Step
    trackbuffer.predict_all()

    # Association Step
    unassigned = trackbuffer.associate_points_to_tracks(pointcloud)
    trackbuffer.update_status()

    # Update Step
    trackbuffer.update_all()

    # Clustering of the remainder points Step
    new_clusters = []
    is_ready = batch_frames(batch, unassigned)
    if is_ready and len(batch.effective_data) != 0:
        new_clusters = apply_DBscan(batch.effective_data)
        batch.empty()

        # Create new track for every new cluster
        trackbuffer.add_tracks(new_clusters)
