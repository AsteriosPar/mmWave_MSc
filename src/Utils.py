from sklearn.cluster import DBSCAN
from collections import deque
import constants as const
import math
import csv
import numpy as np
import os


class RingBuffer:
    """
    Circular buffer with a fixed size that automatically discards the oldest elements
    when new elements are added.

    Attributes
    ----------
    size : int
        Maximum size of the buffer.

    buffer : collections.deque
        Deque representing the circular buffer.

    Methods
    -------
    append(item)
        Add a new element to the buffer. If the buffer is full, the oldest element is removed.

    get_max()
        Get the maximum value in the buffer.

    get_mean()
        Get the mean value of the elements in the buffer.
    """

    def __init__(self, size, init_val=None):
        self.size = size
        self.buffer = deque(maxlen=size)
        if init_val is None:
            self.append(0)
        else:
            self.append(init_val)

    def append(self, item):
        self.buffer.append(item)

    def get_max(self):
        return np.max(self.buffer)

    def get_mean(self):
        return np.mean(self.buffer)


class OfflineManager:
    """
    A class for managing the reading of frames from an offline experiment file.

    Attributes
    ----------
    experiment_path : str
        The path to the directory containing the experiment files.
    frame_count : int
        The count of frames read so far.
    pointer : List[int]
        A list containing two integers: the index of the last read frame and the index of the current file being read.
    pointclouds : Dict[int, Dict[str, List[float]]]
        A dictionary containing point cloud data for each frame.
    last_frame : int or None
        The index of the last frame read from the experiment file, or None if the experiment has finished.

    Methods
    -------
    read_next_frames()
        Read the next batch of frames from the experiment file.
    get_data()
        Get the data for the current frame.
    is_finished() -> bool
        Check if the offline experiment has finished.
    """

    def __init__(self, experiment_path):
        self.experiment_path = experiment_path
        self.frame_count = 0
        self.pointer = [0, 1]
        self.read_next_frames()

    def read_next_frames(self):
        """
        Read the next batch of frames from the given experiment file starting from the specified frame number.
        """
        self.pointclouds = {}
        self.last_frame = None

        while len(self.pointclouds) < const.FB_READ_BUFFER_SIZE:
            file_path = os.path.join(self.experiment_path, f"{self.pointer[1]}.csv")
            try:
                with open(file_path, "r") as file:
                    csv_reader = csv.reader(file)
                    for index, row in enumerate(csv_reader):
                        # Pass previously parsed frames
                        if index < self.pointer[0]:
                            continue

                        framenum = int(row[0])
                        coords = [
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                            float(row[4]),
                            float(row[5]),
                            int(row[6]),
                        ]

                        # Read only the frames in the specified range
                        if framenum in self.pointclouds:
                            # Append coordinates to the existing lists
                            for key, value in zip(
                                ["x", "y", "z", "doppler", "peakVal", "posix"], coords
                            ):
                                self.pointclouds[framenum][key].append(value)
                        else:
                            # If not, create a new dictionary for the framenum
                            self.pointclouds[framenum] = {
                                "x": [coords[0]],
                                "y": [coords[1]],
                                "z": [coords[2]],
                                "doppler": [coords[3]],
                                "peakVal": [coords[4]],
                                "posix": [coords[5]],
                            }

                        self.last_frame = framenum

                        if len(self.pointclouds) >= const.FB_READ_BUFFER_SIZE:
                            # Break the loop once const.FB_READ_BUFFER_SIZE frames are read
                            self.pointer[0] = index + 1
                            break
                    else:
                        self.pointer[0] = 0
                        self.pointer[1] += 1

            except FileNotFoundError:
                break

    def get_data(self):
        """
        Get the data for the current frame.

        Returns
        -------
        exists : bool
            True if data for the current frame exists, False otherwise.
        frame_count : int
            The count of frames read so far.
        data : dict or None
            The point cloud data for the current frame, or None if data for the frame is not available.
        """

        self.frame_count += 1
        # If the read buffer is parsed, read more frames from the experiment file
        if self.frame_count > self.last_frame:
            self.read_next_frames()

        if self.frame_count in self.pointclouds:
            return True, self.frame_count, self.pointclouds[self.frame_count]
        else:
            return False, self.frame_count, None

    def is_finished(self):
        """
        Check if the offline experiment has finished.

        Returns
        -------
        bool
            True if the experiment has finished (last_frame is None), False otherwise.
        """
        return self.last_frame is None


def calc_projection_points(x_origin, y_origin, z_origin):
    """
    Calculate the screen projection of a point based on its distance from a reference point.

    Parameters
    ----------
    x_origin : float
        The reference point coordinate along the x axis.

    y_origin : float
        The reference point coordinate along the y axis.

    z_origin : float
        The reference point coordinate along the z (vertical) axis.

    Returns
    -------
    tuple of floats
        The (x,z) coordinates of the point where the line that connects the reference point
        and the sensitive component of the system, cuts the screen.

    """

    x_dist = x_origin - const.M_X
    y_dist = y_origin - const.M_Y
    z_dist = z_origin - const.M_Z

    if x_dist == 0:
        x_proj = x_origin
    else:
        x1 = -const.M_Y / (y_dist / x_dist)
        x_proj = x1 + const.M_X

    if z_dist == 0:
        z_proj = z_origin
    else:
        z1 = -const.M_Y / (y_dist / z_dist)
        z_proj = z1 + const.M_Z

    return x_proj, z_proj


def altered_EuclideanDist(p1, p2):
    """
    Calculate an altered Euclidean distance between two points in 3D space.

    This distance metric incorporates modifications to better suit the characteristics of cylinder-shaped point clouds,
    especially those representing the human silhouette. It achieves this by applying the following adjustments:

    1. **Vertical Weighting**: Reduces the impact of the vertical distance by using a constant `const.DB_Z_WEIGHT`.
    This is beneficial for improved clustering of cylinder-shaped point clouds.

    2. **Inverse Proportional Weighting**: Introduces a weight to the result inversely proportional to the points' y-axis values.
    This ensures that the distance outputs are lower when the point cloud is further away from the sensor and thus, more sparse.

    Returns
    -------
    float
        The adjusted Euclidean distance between the two points.
    """
    # NOTE: The z-axis has less weight in the distance metric since the sillouette of a person is tall and thin.
    # Also, the further away from the sensor the more sparse the points, so we need a weighing factor
    weight = 1 - ((p1[1] + p2[1]) / 2) * const.DB_RANGE_WEIGHT
    return weight * (
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
        + const.DB_Z_WEIGHT * ((p1[2] - p2[2]) ** 2)
    )


def apply_DBscan(pointcloud, eps=const.DB_EPS, min_samples=const.DB_MIN_SAMPLES_MIN):
    """
    Apply DBSCAN clustering to a 3D point cloud using an altered Euclidean distance metric.

    Parameters
    ----------
    pointcloud : array-like
        The 3D point cloud represented as a list or NumPy array.

    eps : float, optional
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        Default is const.DB_EPS.

    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        Default is const.DB_MIN_SAMPLES.

    Returns
    -------
    list
        A list of clustered point clouds, where each cluster is represented as a list of points.
    """
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
    """
    Transform 3D point coordinates and velocities to a standard axis.

    The transformation includes translation and rotation to bring the input point into a standard coordinate system.

    Parameters
    ----------
    input : array-like
        Input point represented as a 6-element array or list, where the first three elements are coordinates (x, y, z),
        and the last three elements are velocities along the corresponding axes.

    Returns
    -------
    np.array
        Transformed point with coordinates and velocities in the standard axis system.
    """
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


def normalize_data(detObj):
    """
    Preprocesses the point cloud data from the sensor.

    This function filters the input point cloud, converts radial to Cartesian velocity,
    and transforms the coordinates to the standard vertical-horizontal plane axis system.

    Parameters
    ----------
    detObj : dict
        Dictionary containing the raw detection data with keys:
        - "x": x-coordinate
        - "y": y-coordinate
        - "z": z-coordinate
        - "doppler": Doppler velocity
        - "peakVal": Signal Intensity

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
        - doppler
        - peakval
    """

    input_data = np.vstack(
        (detObj["x"], detObj["y"], detObj["z"], detObj["doppler"], detObj["peakVal"])
    ).T
    ef_data = np.empty((0, 8), dtype="float")

    for index in range(len(input_data)):

        # Transform the radial velocity into Cartesian
        r = math.sqrt(
            input_data[index, 0] ** 2
            + input_data[index, 1] ** 2
            + input_data[index, 2] ** 2
        )
        if r == 0:
            vx = 0
            vy = input_data[index, 3]
            vz = 0
        else:
            if (
                input_data[index, 0] is None
                or input_data[index, 1] is None
                or input_data[index, 2] is None
                or input_data[index, 3] is None
            ):
                print(f"Error: {input_data[index, :]}")

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

        transformed_point = np.append(
            transformed_point, (input_data[index, 3], input_data[index, 4])
        )

        # Perform scene constraints filtering
        if (
            transformed_point[2] <= 2.5
            and transformed_point[2] > 0
            and transformed_point[1] > 0
        ):
            ef_data = np.append(
                ef_data,
                [transformed_point],
                axis=0,
            )

    return ef_data


def relative_coordinates(absolute_coords, reference: np.array):
    """
    Calculate relative coordinates on the x and y axes with respect to a reference point.

    Parameters
    ----------
    absolute_coords : np.array
        List of arrays of points in different frames.
    reference : np.array
        Reference point to calculate relative coordinates with respect to.

    Returns
    -------
    np.array
        list of arrays of relative coordinates.
    """

    relative_coords = []
    for frame in absolute_coords:
        relative_coords.append(
            np.array(
                [
                    point - [reference[0], reference[1], 0, 0, 0, 0, 0, 0]
                    for point in frame
                ]
            )
        )

    return relative_coords


def format_single_frame(
    track_cloud, mean=const.INTENSITY_MU, std_dev=const.INTENSITY_STD
):
    """
    Format a single frame of track cloud data.

    This function normalizes the intensity values of the track cloud data, pads or cuts the data to ensure a
    fixed length of 64 points, sorts the data based on the x-coordinate, and reshapes it into an 8x8 matrix.

    Parameters
    ----------
    track_cloud : list
        The list of track cloud data of different frames to be formatted, containing columns for x, y, z, r', intensity.
    mean : float, optional
        The mean value used for intensity normalization (default is const.INTENSITY_MU).
    std_dev : float, optional
        The standard deviation used for intensity normalization (default is const.INTENSITY_STD).

    Returns
    -------
    np.array
        The formatted single frame data reshaped into an 8x8 matrix, with columns for x, y, z, r', intensity.

    """

    sorted_data = np.zeros((const.FB_FRAMES_BATCH + 1, 64, 5))

    for frame_id, frame_cloud in enumerate(track_cloud):

        # print(frame_cloud.shape)
        frame_cloud_final = frame_cloud[:, [0, 1, 2, -2, -1]]
        track_cloud_len = len(frame_cloud_final)

        # Normalize Intensity
        frame_cloud_final[:, 4] = (frame_cloud_final[:, 4] - mean) / std_dev

        # Pad or cut
        if track_cloud_len < 64:
            num_to_pad = 64 - track_cloud_len
            zero_arrays = np.zeros((num_to_pad, 5))
            padded_data = np.concatenate((frame_cloud_final, zero_arrays), axis=0)
        else:
            padded_data = frame_cloud_final[:64]

        # Sort according to x values
        sorted_indices = np.argsort(padded_data[:, 0])
        sorted_data[frame_id] = padded_data[sorted_indices]

    # Resize to matrix (Added a condition two check different models)
    if const.FB_FRAMES_BATCH == 0:
        return sorted_data.reshape((64, 5)).reshape((8, 8, 5))
    else:
        return sorted_data.reshape((const.FB_FRAMES_BATCH + 1, 8, 8, 5))


def format_batched_frames(frame_clouds):

    # We always save three batched frames to the preprocessed files
    full_batch = np.zeros((3 * 64, 5))

    frame_id = 0
    for frame_cloud in reversed(frame_clouds):

        effective_frame_cloud = frame_cloud[:, [0, 1, 2, -2, -1]]
        frame_cloud_len = len(effective_frame_cloud)

        # Pad or cut
        if frame_cloud_len < 64:
            num_to_pad = 64 - frame_cloud_len
            zero_arrays = np.zeros((num_to_pad, 5))
            padded_data = np.concatenate((effective_frame_cloud, zero_arrays), axis=0)
        else:
            padded_data = effective_frame_cloud[:64]

        # Sort
        # sorted_indices = np.argsort(padded_data[:, 0])
        full_batch[frame_id * 64 : (frame_id + 1) * 64] = padded_data
        frame_id += 1

    # Resize to matrix
    return full_batch


def format_single_frame_mode(
    track_cloud: np.array, mean, std_dev, batch_size, fuse=False
):

    track_cloud[:, 4] = (track_cloud[:, 4] - mean) / std_dev

    limited_batch = track_cloud[: 64 * batch_size]

    if fuse:
        non_zero_arrays = limited_batch[~np.all(limited_batch == 0, axis=1)]

        if len(non_zero_arrays) < 64:
            num_to_pad = 64 - len(non_zero_arrays)
            zero_arrays = np.zeros((num_to_pad, 5))
            pl = np.concatenate((non_zero_arrays, zero_arrays), axis=0)
        else:
            pl = non_zero_arrays[:64]

        sorted_indices = np.argsort(pl[:, 0])
        final_frame = pl[sorted_indices]

        return final_frame.reshape((8, 8, 5))
    else:
        return limited_batch.reshape((batch_size, 8, 8, 5))
