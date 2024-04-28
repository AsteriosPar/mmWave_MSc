import numpy as np
import constants as const
import math
import time
from filterpy.kalman import KalmanFilter
from Utils import (
    apply_DBscan,
    RingBuffer,
    relative_coordinates,
    format_single_frame,
)
from typing import List

ACTIVE = 1
INACTIVE = 0

STATIC = True
DYNAMIC = False


class BatchedData(RingBuffer):
    """
    A class to manage and combine frames into a batch.

    Attributes:
    ----------
    - effective_data (numpy.ndarray): An array to store effective data frames.

    Methods:
    -------
    - empty(): Reset the buffer and create an empty effective_data array.
    - add_frame(new_data: numpy.ndarray): Add a new frame of data to the buffer.
    - clear(): Clear the buffer and reset effective_data.
    - change_buffer_size(new_size): Change the size of the buffer.
    - pop_frame(): Remove the oldest frame from the buffer.
    """

    def __init__(self):
        super().__init__(const.FB_FRAMES_BATCH + 1, init_val=np.empty((0, 8)))
        self.effective_data = np.empty((0, 8))

    def add_frame(self, new_data: np.array):
        """
        Add a new frame of data to the buffer.
        """
        while len(self.buffer) >= self.size:
            self.pop_frame()

        super().append(new_data)
        self.effective_data = np.concatenate(list(self.buffer), axis=0)

    def clear(self):
        """
        Clear the buffer and reset effective_data.
        """
        self.buffer.clear()
        self.effective_data = np.array([])

    def change_buffer_size(self, new_size):
        """
        Change the size of the buffer.
        """
        self.size = new_size

    def pop_frame(self):
        """
        Remove the oldest frame from the buffer.
        """
        if len(self.buffer) > 0:
            self.buffer.popleft()


class KalmanState(KalmanFilter):
    """
    A class representing the state of a Kalman filter for motion tracking.

    Attributes:
    ----------
    - centroid: The centroid of the track used for initializing this Kalman filter instance.

    Methods:
    -------
    - __init__(centroid: np.ndarray): Initialize the Kalman filter with default parameters based on the centroid.
    """

    def __init__(self, centroid: np.ndarray):
        super().__init__(
            dim_x=const.MOTION_MODEL.KF_DIM[0], dim_z=const.MOTION_MODEL.KF_DIM[1]
        )

        self.F = const.MOTION_MODEL.KF_F(1)
        self.H = const.MOTION_MODEL.KF_H
        self.Q = const.MOTION_MODEL.KF_Q_DISCR(1)
        self.R = np.eye(const.MOTION_MODEL.KF_DIM[1]) * const.KF_R_STD**2
        self.x = np.array([const.MOTION_MODEL.STATE_VEC(centroid)]).T
        self.P = np.eye(const.MOTION_MODEL.KF_DIM[0]) * const.KF_P_INIT


class PointCluster:
    """
    A class representing a cluster of 3D points and its attributes.

    Attributes:
    ----------
    - pointcloud (numpy.ndarray): An array of 3D points in the form (x, y, z, x', y', z', r', s).
    - point_num (int): The number of points in the cluster.
    - centroid (numpy.ndarray): The centroid of the cluster.
    - min_vals (numpy.ndarray): The minimum values in each dimension of the pointcloud.
    - max_vals (numpy.ndarray): The maximum values in each dimension of the pointcloud.
    - status (bool): The cluster movement status (STATIC: True, DYNAMIC: False)

    Methods:
    -------
    - __init__(pointcloud: numpy.ndarray):
        Initialize PointCluster with a given pointcloud.

    """

    def __init__(self, pointcloud: np.array):
        """
        Initialize PointCluster with a given pointcloud.
        """

        # NOTE: the input is now a list of 8 entries
        self.pointcloud = pointcloud
        self.point_num = pointcloud.shape[0]
        self.centroid = np.mean(pointcloud[:, :6], axis=0)
        self.min_vals = np.min(pointcloud[:, :6], axis=0)
        self.max_vals = np.max(pointcloud[:, :6], axis=0)

        if math.sqrt(np.sum((self.centroid[3:6] ** 2))) < const.TR_VEL_THRES:
            # if pointcloud[6] < const.TR_VEL_THRES:
            self.status = STATIC
        else:
            self.status = DYNAMIC


class ClusterTrack:
    """
    A class representing a tracked cluster with a Kalman filter for motion estimation.

    Parameters
    ----------
    cluster : PointCluster
        The initial point cluster associated with the track.

    Attributes
    ----------
    N_est : int
        Estimated number of points in the cluster.
    spread_est : numpy.ndarray
        Estimated spread of measurements in each dimension.
    group_disp_est : numpy.ndarray
        Estimated group dispersion matrix.
    cluster : PointCluster
        PointCluster associated with the track.
    batch : BatchedData
        The collection of overlaying previous frames
    state : KalmanState
        KalmanState instance for motion estimation.
    status : int (INACTIVE or ACTIVE)
        Current status of the track.
    lifetime : int
        Number of frames the track has been active.
    keypoints : list of floats
        Keypoint x, y, z coordinates of the tracked 19 joints
    color : numpy.ndarray
        Random color assigned to the track for visualization (for visualization purposes).
    predict_x : numpy.ndarray
        Predicted state vector (for visualization purposes).

    Methods
    -------
    predict_state(dt)
        Predict the state of the Kalman filter based on the time multiplier.

    _estimate_point_num()
        Estimate the number of points in the cluster.

    _estimate_measurement_spread()
        Estimate the spread of measurements in each dimension.

    _estimate_group_disp_matrix()
        Estimate the group dispersion matrix.

    _get_D()
        Calculate and get the dispersion matrix for the track.

    associate_pointcloud(pointcloud)
        Associate a new pointcloud with the track.

    get_Rm()
        Get the measurement covariance matrix.

    _get_Rc()
        Get the combined covariance matrix.

    update_state()
        Update the state of the Kalman filter based on the associated pointcloud.

    update_lifetime(reset=False)
        Update the track lifetime.

    seek_inner_clusters()
        Seek inner clusters within the current track.

    """

    def __init__(self, cluster: PointCluster):
        self.N_est = 0
        self.spread_est = np.zeros(const.MOTION_MODEL.KF_DIM[1])
        self.group_disp_est = (
            np.eye(const.MOTION_MODEL.KF_DIM[1]) * const.KF_GROUP_DISP_EST_INIT
        )
        self.cluster = cluster
        self.batch = BatchedData()
        self.state = KalmanState(cluster.centroid)
        self.status = ACTIVE
        self.lifetime = 0
        self.keypoints = const.MODEL_DEFAULT_POSTURE
        # self.height_buffer = RingBuffer(
        #     const.FB_HEIGHT_FRAME_PERIOD, init_val=self.cluster.max_vals[2] - 0.01
        # )
        # self.width_buffer = RingBuffer(const.FB_WIDTH_FRAME_PERIOD)
        # NOTE: For visualizing purposes only
        self.predict_x = self.state.x
        self.color = np.random.rand(
            3,
        )

    def _estimate_point_num(self):
        """
        Estimate the expected number of points in the cluster.
        """
        if const.KF_ENABLE_EST:
            if self.cluster.point_num > self.N_est:
                self.N_est = self.cluster.point_num
            else:
                self.N_est = (
                    1 - const.KF_A_N
                ) * self.N_est + const.KF_A_N * self.cluster.point_num
        else:
            self.N_est = max(const.KF_EST_POINTNUM, self.cluster.point_num)

    def _estimate_measurement_spread(self):
        """
        Estimate the spread of measurements in each dimension.
        """
        for m in range(len(self.cluster.min_vals)):
            spread = self.cluster.max_vals[m] - self.cluster.min_vals[m]

            # Unbiased spread estimation
            if self.cluster.point_num != 1:
                spread = (
                    spread * (self.cluster.point_num + 1) / (self.cluster.point_num - 1)
                )

            # Ensure the computed spread estimation is between 1x and 2x of configured limits
            spread = min(2 * const.KF_SPREAD_LIM[m], spread)
            spread = max(const.KF_SPREAD_LIM[m], spread)

            if spread > self.spread_est[m]:
                self.spread_est[m] = spread
            else:
                self.spread_est[m] = (1.0 - const.KF_A_SPR) * self.spread_est[
                    m
                ] + const.KF_A_SPR * spread

    def _get_D(self):
        """
        Calculate and get the dispersion matrix for the track.

        Returns
        -------
        numpy.ndarray
            Dispersion matrix for the cluster.
        """
        dimension = const.MOTION_MODEL.KF_DIM[1]
        pointcloud = self.cluster.pointcloud
        centroid = self.cluster.centroid
        disp = np.zeros((dimension, dimension), dtype="float")

        for i in range(dimension):
            for j in range(dimension):
                disp[i, j] = np.mean(
                    (pointcloud[:, i] - centroid[i]) * (pointcloud[:, j] - centroid[j])
                )

        return disp

    def _estimate_group_disp_matrix(self):
        """
        Estimate the group dispersion matrix.
        """
        a = self.cluster.point_num / self.N_est
        self.group_disp_est = (1 - a) * self.group_disp_est + a * self._get_D()

    def _get_Rc(self):
        """
        Get the combined covariance matrix.

        Returns
        -------
        numpy.ndarray
            Combined covariance matrix for the cluster.
        """
        N = self.cluster.point_num
        N_est = self.N_est
        return (self.get_Rm() / N) + (
            (N_est - N) / ((N_est - 1) * N)
        ) * self.group_disp_est

    def associate_pointcloud(self, pointcloud: np.array):
        """
        Associate a point cloud with the track.

        Parameters
        ----------
        pointcloud : np.array
            2D NumPy array representing the point cloud.

        Notes
        -----
        This method performs the following steps:
        1. Initializes a PointCluster with the given point cloud.
        2. Adds the point-cluster to the track's frames batch.
        3. Estimates the number of points in the cluster.
        4. Estimates the spread of measurements in the cluster.
        5. Estimates the dispersion matrix of the point groups in the cluster.

        Parameters
        ----------
        pointcloud : np.array
            2D NumPy array representing the point cloud.
        """
        self.cluster = PointCluster(pointcloud)
        self.batch.add_frame(self.cluster.pointcloud)
        self._estimate_point_num()
        self._estimate_measurement_spread()
        self._estimate_group_disp_matrix()

        # Save the current height and width of the pointcloud projection to the screen in the ringbuffers.
        # TODO: This approach needs to change
        # self.height_buffer.append(
        #     calc_projection_points(
        #         value=self.cluster.max_vals[2] - 0.01,
        #         y=self.cluster.min_vals[1],
        #         vertical_axis=True,
        #     )
        # )
        # self.width_buffer.append(
        #     calc_projection_points(
        #         value=self.cluster.max_vals[0], y=self.cluster.min_vals[1]
        #     )
        #     - calc_projection_points(
        #         value=self.cluster.min_vals[0], y=self.cluster.min_vals[1]
        #     )
        # )

    def get_Rm(self):
        """
        Get the measurement covariance matrix

        Returns
        -------
        numpy.ndarray
            Measurement covariance matrix for the cluster.
        """
        return np.diag(((self.spread_est / 2) ** 2))

    def predict_state(self, dt: float):
        """
        Predict the state of the Kalman filter based on the time multiplier.

        Parameters
        ----------
        dt : float
            Time multiplier for the prediction.
        """
        self.state.predict(
            F=const.MOTION_MODEL.KF_F(dt),
            Q=const.MOTION_MODEL.KF_Q_DISCR(dt),
        )
        self.predict_x = self.state.x

    def update_state(self):
        """
        Update the state of the Kalman filter based on the associated measurement (pointcloud centroid).
        """
        z = np.array(self.cluster.centroid)
        x_prev = self.state.x[:2, 0]
        self.state.update(z, R=self._get_Rc())

        # If the variance between the predicted and measured position
        variance = z[:1] - self.state.x[:1, 0]
        if abs(variance.any()) > 0.6 and self.lifetime == 0:
            self.state.x[:1, 0] += variance * 0.4

    def update_lifetime(self, dt, reset=False):
        """
        Update the track lifetime.
        """
        if reset:
            self.lifetime = 0
        else:
            self.lifetime += dt

    def seek_inner_clusters(self):
        """
        Seek inner clusters within the current cluster.

        This method uses DBSCAN to identify inner clusters within the current cluster's pointcloud.
        It helps in separating inner clusters and filtering noise when a single cluster is detected.

        Returns
        -------
        list
            List of new inner track clusters (PointCluster instances).

        """

        new_track_clusters = []
        spread = self.cluster.max_vals[:1] - self.cluster.min_vals[:1]
        if (
            self.cluster.point_num > const.DB_POINTS_THRES
            and spread.any() > const.DB_SPREAD_THRES
        ):
            # Allow frame fusion for better resolution
            # NOTE: State machine is better here
            if self.cluster.status == STATIC:
                self.batch.change_buffer_size(const.FB_FRAMES_BATCH_STATIC)
            else:
                self.batch.change_buffer_size(const.FB_FRAMES_BATCH)

            self.batch.add_frame(self.cluster.pointcloud)
            pointcloud = self.batch.effective_data

            # Apply clustering to identify inner clusters
            track_clusters = apply_DBscan(
                pointcloud=pointcloud,
                eps=const.DB_INNER_EPS,
            )

            if len(track_clusters) > 1:
                new_track_clusters = [track_clusters[1]]

        return new_track_clusters


class TrackBuffer:
    """
    A class representing a buffer for managing and updating the multiple ClusterTracks of the scene.

    Attributes
    ----------
    effective_tracks : List[ClusterTrack]
        List of currently active (non-INACTIVE) tracks in the buffer.
    next_track_id : int
        The id int that will be given to the next active track.
    dt : float
        Time multiplier used for predicting states. Indicates the time passed since the previous
        valid observed frame.
    t : float
        Current time when the TrackBuffer is instantiated / updated.

    Methods
    -------
    _maintain_tracks()
        Update the status of tracks based on their lifetime.

    update_ef_tracks()
        Update the list of effective tracks (excluding INACTIVE tracks).

    has_active_tracks()
        Check if there are active tracks in the buffer.

    _calc_dist_fun(full_set)
        Calculate the Mahalanobis distance matrix for gating.

    _add_tracks(new_clusters)
        Add new tracks to the buffer.

    _predict_all()
        Predict the state of all effective tracks.

    _update_all()
        Update the state of all effective tracks.

    _get_gated_clouds(full_set)
        Gate the pointcloud and return gated and unassigned clouds.

    _associate_points_to_tracks(full_set)
        Associate points to existing tracks and handle inner cluster separation.

    track(pointcloud, batch)
        Perform the tracking process including prediction, association, status update, and clustering.

    estimate_posture(model)
        Estimate the posture of each track in the buffer using a CNN model.

    """

    def __init__(self):
        """
        Initialize TrackBuffer with empty lists for tracks and effective tracks.
        """
        self.effective_tracks: List[ClusterTrack] = []
        self.next_track_id = 0
        self.dt = 0
        self.t = time.time()

    def _maintain_tracks(self):
        """
        Update the status of tracks based on their mobility and lifetime. Then update the list of effective tracks.
        """
        for track in self.effective_tracks:
            if track.cluster.status == DYNAMIC:
                lifetime = const.TR_LIFETIME_DYNAMIC
            else:
                lifetime = const.TR_LIFETIME_STATIC

            if track.lifetime > lifetime:
                track.status = INACTIVE

        self.effective_tracks[:] = [
            track for track in self.effective_tracks if track.status != INACTIVE
        ]

    def _calc_dist_fun(self, full_set: np.array):
        """
        Calculate the Mahalanobis distance matrix for gating.

        Parameters
        ----------
        full_set : np.ndarray
            Full set of points.

        Returns
        -------
        np.ndarray
            An array representing the associated track (entry) for each point (index).
            If no track is associated with a point, the entry is set to None.
        """
        dist_matrix = np.empty((full_set.shape[0], len(self.effective_tracks)))
        associated_track_for = np.full(full_set.shape[0], None, dtype=object)

        for j, track in enumerate(self.effective_tracks):
            H_i = np.dot(const.MOTION_MODEL.KF_H, track.state.x).flatten()
            # Group residual covariance matrix
            C_g_j = track.state.P[:6, :6] + track.get_Rm() + track.group_disp_est

            for i, point in enumerate(full_set):
                # Innovation for each measurement
                y_ij = np.array(point[:6]) - H_i

                # Distance function (d^2)
                dist_matrix[i][j] = np.log(np.abs(np.linalg.det(C_g_j))) + np.dot(
                    np.dot(y_ij.T, np.linalg.inv(C_g_j)), y_ij
                )

                # Perform Gate threshold check
                if dist_matrix[i][j] < const.TR_GATE:
                    # Just choose the closest mahalanobis distance
                    if associated_track_for[i] is None:
                        associated_track_for[i] = j
                    else:
                        if (
                            dist_matrix[i][j]
                            < dist_matrix[i][int(associated_track_for[i])]
                        ):
                            associated_track_for[i] = j

        return associated_track_for

    def _add_tracks(self, new_clusters):
        """
        Add new tracks to the buffer.

        Parameters
        ----------
        new_clusters : list
            List of new clusters to be added as tracks.
        """
        for new_cluster in new_clusters:
            new_track = ClusterTrack(PointCluster(np.array(new_cluster)))
            # new_track.id = self.next_track_id
            self.next_track_id += 1
            self.effective_tracks.append(new_track)

    def _predict_all(self):
        """
        Predict the state of all effective tracks.
        """
        for track in self.effective_tracks:
            track.predict_state(track.lifetime + self.dt)

    def _update_all(self):
        """
        Update the state of all effective tracks.
        """
        for track in self.effective_tracks:
            track.update_state()

    def _get_gated_clouds(self, full_set: np.array):
        """
        Split the pointcloud according to the formed gates and return gated and unassigned clouds.

        Parameters
        ----------
        full_set : np.array
            Full set of points.

        Returns
        -------
        tuple
            Tuple containing unassigned points and clustered clouds.
        """
        unassigned = np.empty((0, 8), dtype="float")
        clusters = [[] for _ in range(len(self.effective_tracks))]
        # Simple matrix has len = len(full_set) and has the index of the chosen track.
        associated_track_for = self._calc_dist_fun(full_set)

        for i, point in enumerate(full_set):
            if associated_track_for[i] is None:
                unassigned = np.append(unassigned, [point], axis=0)
            else:
                clusters[associated_track_for[i]].append(point)
        return unassigned, clusters

    def _associate_points_to_tracks(self, full_set: np.array):
        """
        Associate points to existing tracks and handle inner cluster separation.

        Parameters
        ----------
        full_set : np.array
            Full set of sensed points.

        Returns
        -------
        np.ndarray
            Unassigned points.
        """
        unassigned, clouds = self._get_gated_clouds(full_set)
        new_inner_clusters = []

        for j, track in enumerate(self.effective_tracks):
            if len(clouds[j]) == 0:
                track.update_lifetime(dt=self.dt)
            else:
                track.update_lifetime(dt=self.dt, reset=True)
                track.associate_pointcloud(np.array(clouds[j]))

                # inner cluster separation
                # new_inner_clusters.append(track.seek_inner_clusters())

        # In case inner clusters are found, create new tracks for them
        for inner_cluster in new_inner_clusters:
            self._add_tracks(inner_cluster)

        return unassigned

    def track(self, pointcloud, batch: BatchedData):
        """
        Perform the tracking process including prediction, association, maintenance, update, and clustering.

        Parameters
        ----------
        pointcloud : np.array
            Pointcloud data.
        batch : BatchedData
            BatchedData instance for managing frames.

        Returns
        -------
        None
        """
        # Prediction Step
        self._predict_all()

        # Association Step
        unassigned = self._associate_points_to_tracks(pointcloud)
        self._maintain_tracks()

        # Update Step
        self._update_all()

        # Clustering of the remainder points Step
        new_clusters = []
        batch.add_frame(unassigned)

        if (
            len(batch.effective_data) > 0
            and len(self.effective_tracks) < const.TR_MAX_TRACKS
        ):
            new_clusters = apply_DBscan(batch.effective_data)

            if len(new_clusters) > 0:
                batch.clear()

            # Create new track for every new cluster
            self._add_tracks(new_clusters)

    def estimate_posture(self, model):
        """
        Format the pointcloud, estimate and save the posture of the target of each track using a CNN model.

        Parameters
        ----------
        model : Model
            The CNN model used for posture estimation.

        Returns
        -------
        None
        """
        frame_matrices = []
        indexes = []
        for index, track in enumerate(self.effective_tracks):
            if len(track.batch.effective_data) > const.MODEL_MIN_INPUT:
                rel_track_points = relative_coordinates(
                    track.batch.effective_data,
                    track.cluster.centroid[:2],
                )
                # The inputs are in the form of [x, y, z, x', y', z', r', s]
                frame_matrices.append(
                    format_single_frame(rel_track_points[:, [0, 1, 2, -2, -1]])
                )
                indexes.append(index)

        frame_matrices_array = np.array(frame_matrices)
        if len(frame_matrices_array) > 0:
            frame_keypoints = model.predict(frame_matrices_array)
            for i, index in enumerate(indexes):
                self.effective_tracks[index].keypoints = frame_keypoints[i]
