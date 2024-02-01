import numpy as np
import constants as const
import math
import time
from filterpy.kalman import KalmanFilter
from utils import apply_DBscan, RingBuffer, calc_projection_points
from typing import List

ACTIVE = 1
INACTIVE = 0

STATIC = True
DYNAMIC = False


class BatchedData:
    """
    A class to manage and combine frames into a batch for a more resourceful analysis.

    Attributes:
    ----------
    - counter (int): A counter to keep track of the number of frames added.
    - effective_data (numpy.ndarray): An array to store effective data frames.

    Methods:
    -------
    - empty(): Reset the counter and create an empty effective_data array.
    - add_frame(new_data: numpy.ndarray): Add a new frame of data to the effective_data array.
    - is_complete() -> bool: Check if the batch is complete
    """

    def __init__(self):
        self.counter = 0
        self.effective_data = np.empty((0, const.MOTION_MODEL.KF_DIM[1]), dtype="float")

    def empty(self):
        """
        Reset the counter and create an empty effective_data array
        """
        self.counter = 0
        self.effective_data = np.empty((0, const.MOTION_MODEL.KF_DIM[1]), dtype="float")

    def add_frame(self, new_data: np.array):
        """
        Add a new frame of data to the effective_data array.
        """
        self.effective_data = np.append(self.effective_data, new_data, axis=0)
        self.counter += 1

    def is_complete(self):
        """
        Check if the batch is complete based on the counter and the threshold FB_FRAMES_BATCH
        set in the constants.py file.

        Returns
        -------
        bool
            Whether or not the batch is complete.
        """
        return self.counter >= (const.FB_FRAMES_BATCH - 1)


class KalmanState:
    """
    A class representing the state of a Kalman filter for motion tracking.

    Attributes:
    ----------
    - inst (filterpy.kalman.KalmanFilter): The Kalman filter instance.
    - centroid: The centroid of the track used for initializing this Kalman filter instance.

    Note: In this class there is no control function B

    """

    def __init__(self, centroid):
        self.inst = KalmanFilter(
            dim_x=const.MOTION_MODEL.KF_DIM[0],
            dim_z=const.MOTION_MODEL.KF_DIM[1],
        )
        self.inst.F = const.MOTION_MODEL.KF_F(1)
        self.inst.H = const.MOTION_MODEL.KF_H
        self.inst.Q = const.MOTION_MODEL.KF_Q_DISCR(1)
        # We assume independent noise in the x,y,z variables of equal standard deviations.
        self.inst.R = np.eye(const.MOTION_MODEL.KF_DIM[1]) * const.KF_R_STD**2
        # For initial values
        self.inst.x = np.array([const.MOTION_MODEL.STATE_VEC(centroid)]).T
        self.inst.P = np.eye(const.MOTION_MODEL.KF_DIM[0]) * const.KF_P_INIT


class PointCluster:
    """
    A class representing a cluster of 3D points and its attributes.

    Attributes:
    ----------
    - pointcloud (numpy.ndarray): An array of 3D points in the form (x, y, z).
    - point_num (int): The number of points in the cluster.
    - centroid (numpy.ndarray): The centroid of the cluster.
    - min_vals (numpy.ndarray): The minimum values in each dimension of the pointcloud.
    - max_vals (numpy.ndarray): The maximum values in each dimension of the pointcloud.
    - status (bool): The cluster movement status (STATIC: True, DYNAMIC: False)

    Methods:
    -------
    - __init__(pointcloud: numpy.ndarray):
        Initialize PointCluster with a given pointcloud.

    Note: Assumes that pointcloud is an np.array of tuples (x, y, z, x', y', z').

    """

    def __init__(self, pointcloud: np.array):
        """
        Initialize PointCluster with a given pointcloud.
        """
        self.pointcloud = pointcloud
        self.point_num = pointcloud.shape[0]
        self.centroid = np.mean(pointcloud, axis=0)
        self.min_vals = np.min(pointcloud, axis=0)
        self.max_vals = np.max(pointcloud, axis=0)

        if math.sqrt(np.sum((self.centroid[3:] ** 2))) < const.TR_VEL_THRES:
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
    id : int or None
        Identifier for the track.
    N_est : int
        Estimated number of points in the cluster.
    spread_est : numpy.ndarray
        Estimated spread of measurements in each dimension.
    group_disp_est : numpy.ndarray
        Estimated group dispersion matrix.
    cluster : PointCluster
        PointCluster associated with the track.
    batch : BatcedData
        The collection of overlaying previous frames
    state : KalmanState
        KalmanState instance for motion estimation.
    status : int (INACTIVE or ACTIVE)
        Current status of the track.
    lifetime : int
        Number of frames the track has been active.
    color : numpy.ndarray
        Random color assigned to the track for visualization.
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

    associate_pointcloud(pointcloud)
        Associate a new pointcloud with the track.

    get_Rm()
        Get the measurement covariance matrix.

    get_Rc()
        Get the combined covariance matrix.

    update_state()
        Update the state of the Kalman filter based on the associated pointcloud.

    update_lifetime(reset=False)
        Update the track lifetime.

    seek_inner_clusters()
        Seek inner clusters within the current track.

    """

    def __init__(self, cluster: PointCluster):
        self.id = None
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
        self.color = np.random.rand(
            3,
        )
        self.height_buffer = RingBuffer(
            const.FB_HEIGHT_FRAME_PERIOD, init_val=self.cluster.max_vals[2] - 0.01
        )
        self.width_buffer = RingBuffer(const.FB_WIDTH_FRAME_PERIOD)
        # NOTE: For visualizing purposes only
        self.predict_x = self.state.inst.x

    def predict_state(self, dt: float):
        """
        Predict the state of the Kalman filter based on the time multiplier.

        Parameters
        ----------
        dt : float
            Time multiplier for the prediction.
        """
        self.state.inst.predict(
            F=const.MOTION_MODEL.KF_F(dt),
            Q=const.MOTION_MODEL.KF_Q_DISCR(dt),
        )
        self.predict_x = self.state.inst.x

    def _estimate_point_num(self):
        """
        Estimate the number of points in the cluster.
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

    def get_D(self):
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
        self.group_disp_est = (1 - a) * self.group_disp_est + a * self.get_D()

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
        2. Estimates the number of points in the cluster.
        3. Estimates the spread of measurements in the cluster.
        4. Estimates the dispersion matrix of the point groups in the cluster.
        5. Saves the current height and width of the point cloud projection to the screen in the ring buffers.

        Parameters
        ----------
        pointcloud : np.array
            2D NumPy array representing the point cloud.
        """
        self.cluster = PointCluster(pointcloud)
        self._estimate_point_num()
        self._estimate_measurement_spread()
        self._estimate_group_disp_matrix()

        # Save the current height and width of the pointcloud projection to the screen in the ringbuffers.
        self.height_buffer.append(
            calc_projection_points(
                value=self.cluster.max_vals[2] - 0.01,
                y=self.cluster.min_vals[1],
                vertical_axis=True,
            )
        )
        self.width_buffer.append(
            calc_projection_points(
                value=self.cluster.max_vals[0], y=self.cluster.min_vals[1]
            )
            - calc_projection_points(
                value=self.cluster.min_vals[0], y=self.cluster.min_vals[1]
            )
        )

    def get_Rm(self):
        """
        Get the measurement covariance matrix

        Returns
        -------
        numpy.ndarray
            Measurement covariance matrix for the cluster.
        """
        return np.diag(((self.spread_est / 2) ** 2))

    def get_Rc(self):
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

    def update_state(self):
        """
        Update the state of the Kalman filter based on the associated measurement (pointcloud centroid).
        """
        self.state.inst.update(np.array(self.cluster.centroid), R=self.get_Rc())

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

        # Allow frame overlaying over static tracks for better resolution
        if self.cluster.status == STATIC:
            self.batch.add_frame(self.cluster.pointcloud)
            pointcloud = self.batch.effective_data
        else:
            pointcloud = self.cluster.pointcloud

        # Apply clustering to identify inner clusters
        track_clusters = apply_DBscan(
            pointcloud=pointcloud,
            eps=const.DB_INNER_EPS,
            min_samples=const.DB_INNER_MIN_SAMPLES,
        )
        new_track_clusters = []

        # NOTE: This might work for filtering noise now, but might be problematic if kept when feeding the AI network
        # if len(track_clusters) > 0:
        # self.associate_pointcloud(np.array(track_clusters[0]))

        if len(track_clusters) > 1:
            new_track_clusters = [track_clusters[1]]
            self.batch.empty()

        if self.cluster.status == DYNAMIC or self.batch.is_complete():
            self.batch.empty()

        return new_track_clusters


class TrackBuffer:
    """
    A class representing a buffer for managing and updating the multiple ClusterTracks of the scene.

    Attributes
    ----------
    tracks : List[ClusterTrack]
        List of all tracks in the buffer.
    effective_tracks : List[ClusterTrack]
        List of currently active (non-INACTIVE) tracks in the buffer.
    dt : float
        Time multiplier used for predicting states. Indicates the time passed since the previous
        valid observed frame.
    t : float
        Current time when the TrackBuffer is instantiated / updated.

    Methods
    -------
    update_status()
        Update the status of tracks based on their lifetime.

    update_ef_tracks()
        Update the list of effective tracks (excluding INACTIVE tracks).

    has_active_tracks()
        Check if there are active tracks in the buffer.

    _calc_dist_fun(full_set)
        Calculate the Mahalanobis distance matrix for gating.

    add_tracks(new_clusters)
        Add new tracks to the buffer.

    predict_all()
        Predict the state of all effective tracks.

    update_all()
        Update the state of all effective tracks.

    get_gated_clouds(full_set)
        Gate the pointcloud and return gated and unassigned clouds.

    associate_points_to_tracks(full_set)
        Associate points to existing tracks and handle inner cluster separation.

    track(pointcloud, batch)
        Perform the tracking process including prediction, association, status update, and clustering.
    """

    def __init__(self):
        """
        Initialize TrackBuffer with empty lists for tracks and effective tracks.
        """
        self.tracks: List[ClusterTrack] = []
        self.effective_tracks: List[ClusterTrack] = []
        self.dt = 0
        self.t = time.time()

    def update_status(self):
        """
        Update the status of tracks based on their mobility and lifetime.
        """
        for track in self.effective_tracks:
            if track.cluster.status == DYNAMIC:
                lifetime = const.TR_LIFETIME_DYNAMIC
            else:
                lifetime = const.TR_LIFETIME_STATIC

            if track.lifetime > lifetime:
                track.status = INACTIVE

    def update_ef_tracks(self):
        """
        Update the list of effective tracks.
        """
        self.effective_tracks = [
            track for track in self.tracks if track.status != INACTIVE
        ]

    def has_active_tracks(self):
        """
        Check if there are active tracks in the buffer.

        Returns
        -------
        bool
            True if there are active tracks, False otherwise.
        """
        if len(self.effective_tracks) != 0:
            return True
        else:
            return False

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
        dist_matrix = np.empty((full_set.shape[0], len(self.tracks)))
        associated_track_for = np.full(full_set.shape[0], None, dtype=object)

        for track in self.effective_tracks:
            j = track.id
            H_i = np.dot(const.MOTION_MODEL.KF_H, track.state.inst.x).flatten()
            # Group residual covariance matrix
            C_g_j = track.state.inst.P[:6, :6] + track.get_Rm() + track.group_disp_est

            for i, point in enumerate(full_set):
                # Innovation for each measurement
                y_ij = np.array(point) - H_i

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

    def add_tracks(self, new_clusters):
        """
        Add new tracks to the buffer.

        Parameters
        ----------
        new_clusters : list
            List of new clusters to be added as tracks.
        """
        for new_cluster in new_clusters:
            new_track = ClusterTrack(PointCluster(np.array(new_cluster)))
            new_track.id = len(self.tracks)
            self.tracks.append(new_track)
            self.effective_tracks.append(new_track)

    def predict_all(self):
        """
        Predict the state of all effective tracks.
        """
        for track in self.effective_tracks:
            track.predict_state(track.lifetime + self.dt)

    def update_all(self):
        """
        Update the state of all effective tracks.
        """
        for track in self.effective_tracks:
            track.update_state()

    def get_gated_clouds(self, full_set: np.array):
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
        unassigned = np.empty((0, const.MOTION_MODEL.KF_DIM[1]), dtype="float")
        clusters = [[] for _ in range(len(self.effective_tracks))]
        # Simple matrix has len = len(full_set) and has the index of the chosen track.
        associated_track_for = self._calc_dist_fun(full_set)

        for i, point in enumerate(full_set):
            if associated_track_for[i] is None:
                unassigned = np.append(unassigned, [point], axis=0)
            else:
                list_index = None
                for index, track in enumerate(self.effective_tracks):
                    if track.id == associated_track_for[i]:
                        list_index = index
                        break

                clusters[list_index].append(point)
        return unassigned, clusters

    def associate_points_to_tracks(self, full_set: np.array):
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
        unassigned, clouds = self.get_gated_clouds(full_set)
        new_inner_clusters = []

        for j, track in enumerate(self.effective_tracks):
            if len(clouds[j]) == 0:
                track.update_lifetime(dt=self.dt)
            else:
                track.update_lifetime(dt=self.dt, reset=True)
                track.associate_pointcloud(np.array(clouds[j]))
                # inner cluster separation
                new_inner_clusters.append(track.seek_inner_clusters())

        # In case inner clusters are found, create new tracks for them
        for inner_cluster in new_inner_clusters:
            self.add_tracks(inner_cluster)

        return unassigned

    def track(self, pointcloud, batch: BatchedData):
        """
        Perform the tracking process including prediction, association, status update, and clustering.

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
        self.predict_all()

        # Association Step
        unassigned = self.associate_points_to_tracks(pointcloud)
        self.update_status()
        self.update_ef_tracks()

        # Update Step
        self.update_all()

        # Clustering of the remainder points Step
        new_clusters = []
        batch.add_frame(unassigned)
        if len(batch.effective_data) > 0:
            new_clusters = apply_DBscan(batch.effective_data)

            if batch.is_complete or len(new_clusters) > 0:
                batch.empty()

            # Create new track for every new cluster
            self.add_tracks(new_clusters)
