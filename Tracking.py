import numpy as np
import constants as const
from filterpy.kalman import KalmanFilter
from utils import apply_DBscan
from typing import List

DETECTED = 2
ACTIVE = 1
INACTIVE = 0


class BatchedData:
    def __init__(self):
        self.counter = 0
        self.effective_data = np.empty((0, const.MOTION_MODEL.KF_DIM[1]), dtype="float")

    def empty(self):
        self.counter = 0
        self.effective_data = np.empty((0, const.MOTION_MODEL.KF_DIM[1]), dtype="float")

    def add_frame(self, new_data: np.array):
        self.effective_data = np.append(self.effective_data, new_data, axis=0)
        self.counter += 1

    def is_complete(self):
        return self.counter >= (const.FB_FRAMES_BATCH - 1)


class KalmanState:
    def __init__(self, centroid):
        # No control function
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
    def get_min_max_coords(self, pointcloud):
        min_values = np.min(pointcloud, axis=0)
        max_values = np.max(pointcloud, axis=0)
        return (min_values, max_values)

    def __init__(self, pointcloud: np.array):
        # pointcloud should be an np.array of tuples (x,y,z)
        self.pointcloud = pointcloud
        self.point_num = pointcloud.shape[0]
        self.centroid = np.mean(pointcloud, axis=0)
        (self.min_vals, self.max_vals) = self.get_min_max_coords(pointcloud)


class ClusterTrack:
    def __init__(self, cluster: PointCluster):
        self.id = None
        self.N_est = 0
        self.spread_est = np.zeros(const.MOTION_MODEL.KF_DIM[1])
        self.group_disp_est = (
            np.eye(const.MOTION_MODEL.KF_DIM[1]) * const.KF_GROUP_DISP_EST_INIT
        )
        self.cluster = cluster
        self.state = KalmanState(cluster.centroid)
        self.status = ACTIVE
        self.lifetime = 0
        self.det_lifetime = 0
        self.undetected_dt = 0
        self.color = np.random.rand(
            3,
        )
        # NOTE: For visualizing purposes only
        self.predict_x = self.state.inst.x

    def predict_state(self, dt_multiplier):
        self.state.inst.predict(
            F=const.MOTION_MODEL.KF_F(dt_multiplier),
            Q=const.MOTION_MODEL.KF_Q_DISCR(dt_multiplier),
        )
        self.predict_x = self.state.inst.x

    def _estimate_point_num(self):
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
        a = self.cluster.point_num / self.N_est
        self.group_disp_est = (1 - a) * self.group_disp_est + a * self.get_D()

    def associate_pointcloud(self, pointcloud: np.array):
        self.cluster = PointCluster(pointcloud)
        self._estimate_point_num()
        self._estimate_measurement_spread()
        self._estimate_group_disp_matrix()

    def get_Rm(self):
        return np.diag(((self.spread_est / 2) ** 2))

    def get_Rc(self):
        N = self.cluster.point_num
        N_est = self.N_est
        return (self.get_Rm() / N) + (
            (N_est - N) / ((N_est - 1) * N)
        ) * self.group_disp_est

    def update_state(self):
        self.state.inst.update(np.array(self.cluster.centroid), R=self.get_Rc())

    def update_lifetime(self, reset=False):
        if reset:
            self.lifetime = 0
        else:
            self.lifetime += 1

        if self.status == DETECTED:
            self.det_lifetime += 1

    def update_dt(self, reset=False):
        if reset:
            self.undetected_dt = 0
        else:
            self.undetected_dt += self.undetected_dt

    def seek_inner_clusters(self):
        # NOTE: Apart from separating inner clusters, this module
        # helps filter noise in case a single cluster is detected
        track_clusters = apply_DBscan(self.cluster.pointcloud)
        new_track_clusters = []
        if len(track_clusters) == 1:
            self.associate_pointcloud(np.array(track_clusters[0]))

        elif len(track_clusters) > 1:
            # print(track_clusters)
            if self.status == DETECTED:
                self.status = ACTIVE
                self.associate_pointcloud(np.array(track_clusters[0]))
                new_track_clusters = [track_clusters[1]]
            else:
                self.status = DETECTED

        return new_track_clusters


class TrackBuffer:
    def __init__(self):
        self.tracks: List[ClusterTrack] = []
        self.effective_tracks: List[ClusterTrack] = []
        self.dt_multiplier = 1

    def update_status(self):
        for track in self.effective_tracks:
            if track.lifetime > const.TR_LIFETIME:
                track.status = INACTIVE
            elif track.det_lifetime > const.TR_LIFETIME_DET:
                # Removes the DETECTED state off the track
                track.status = ACTIVE
                track.det_lifetime = 0

    def update_ef_tracks(self):
        self.effective_tracks = [
            track for track in self.tracks if track.status != INACTIVE
        ]

    def has_active_tracks(self):
        if len(self.effective_tracks) != 0:
            return True
        else:
            return False

    def _calc_dist_fun(self, full_set: np.array):
        dist_matrix = np.empty((full_set.shape[0], len(self.tracks)))
        simple_approach = np.full(full_set.shape[0], None, dtype=object)

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
                    if simple_approach[i] is None:
                        simple_approach[i] = j
                    else:
                        if dist_matrix[i][j] < dist_matrix[i][int(simple_approach[i])]:
                            simple_approach[i] = j

        # return dist_matrix
        return simple_approach

    def add_tracks(self, new_clusters):
        for new_cluster in new_clusters:
            new_track = ClusterTrack(PointCluster(np.array(new_cluster)))
            new_track.id = len(self.tracks)
            self.tracks.append(new_track)
            self.effective_tracks.append(new_track)

    def predict_all(self):
        for track in self.effective_tracks:
            track.predict_state(track.undetected_dt + self.dt_multiplier)

    def update_all(self):
        for track in self.effective_tracks:
            track.update_state()

    def get_gated_clouds(self, full_set: np.array):
        unassigned = np.empty((0, const.MOTION_MODEL.KF_DIM[1]), dtype="float")
        clusters = [[] for _ in range(len(self.effective_tracks))]
        # Simple matrix has len = len(full_set) and has the index of the chosen track.
        simple_matrix = self._calc_dist_fun(full_set)

        for i, point in enumerate(full_set):
            if simple_matrix[i] is None:
                unassigned = np.append(unassigned, [point], axis=0)
            else:
                list_index = None
                for index, track in enumerate(self.effective_tracks):
                    if track.id == simple_matrix[i]:
                        list_index = index
                        break

                clusters[list_index].append(point)
        return unassigned, clusters

    def associate_points_to_tracks(self, full_set: np.array):
        unassigned, clouds = self.get_gated_clouds(full_set)
        new_inner_clusters = []

        for j, track in enumerate(self.effective_tracks):
            if len(clouds[j]) == 0:
                track.update_lifetime()
                track.update_dt()
            else:
                track.update_lifetime(reset=True)
                track.update_dt(reset=True)
                track.associate_pointcloud(np.array(clouds[j]))
                # inner cluster separation
                new_inner_clusters.append(track.seek_inner_clusters())

        # In case inner clusters are found, create new tracks for them
        for inner_cluster in new_inner_clusters:
            self.add_tracks(inner_cluster)

        return unassigned

    def track(self, pointcloud, batch: BatchedData):
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
        if batch.is_complete and len(batch.effective_data) > 0:
            new_clusters = apply_DBscan(batch.effective_data)
            batch.empty()

            # Create new track for every new cluster
            self.add_tracks(new_clusters)
