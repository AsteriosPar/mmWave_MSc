import numpy as np
import constants as const
from filterpy.kalman import KalmanFilter
from Localization import apply_DBscan, apply_constraints


class KalmanState:
    def __init__(self, centroid):
        # My state variables are: [x y z x' y' z']
        # My input variables are: [x y z]
        # No control function
        self.inst = KalmanFilter(dim_x=6, dim_z=3)

        self.inst.F = const.EKF_F(1)
        self.inst.H = const.EKF_H

        # We assume independent noise in the x,y,z variables of equal standard deviations.
        self.inst.Q = const.EKF_Q_DISCR(1)
        self.inst.R = np.eye(3) * const.EKF_R_STD**2

        # For initial values
        self.inst.x = np.array([[centroid[0], centroid[1], centroid[2], 0, 0, 0]]).T
        self.inst.P = np.eye(6) * 100.0


class PointCluster:
    def calc_centroid(self, pointcloud):
        return np.mean(pointcloud, axis=0)

    def get_min_max_coords(self, pointcloud):
        min_values = np.min(pointcloud, axis=0)
        max_values = np.max(pointcloud, axis=0)
        return (min_values, max_values)

    def __init__(self, pointcloud: np.array):
        # pointcloud should be an np.array of tuples (x,y,z)
        self.pointcloud = pointcloud
        self.point_num = pointcloud.shape[0]
        self.centroid = self.calc_centroid(pointcloud)
        (self.min_vals, self.max_vals) = self.get_min_max_coords(pointcloud)


class ClusterTrack:
    def __init__(self, cluster: PointCluster):
        self.id = None
        # Number of previously estimated points
        self.N_est = None
        self.spread_est = np.array([0, 0, 0])
        self.cluster = cluster
        self.state = KalmanState(cluster.centroid)
        self.status = 1
        self.lifetime = 0
        self.color = np.random.rand(
            3,
        )

    def predict_state(self, dt_multiplier):
        self.state.inst.predict(
            F=const.EKF_F(dt_multiplier), Q=const.EKF_Q_DISCR(dt_multiplier)
        )

    def _estimate_point_num(self, enable=False):
        if enable:
            if self.cluster.point_num > self.N_est:
                self.N_est = self.cluster.point_num
            else:
                self.N_est = (
                    1 - const.EKF_A_N
                ) * self.N_est + const.EKF_A_N * self.cluster.point_num
        else:
            self.N_est = max(const.EKF_EST_POINTNUM, self.cluster.point_num)

    def _estimate_measurement_spread(self):
        for m in range(len(self.cluster.min_vals)):
            spread = self.cluster.max_vals[m] - self.cluster.min_vals[m]

            # Unbiased spread estimation
            # This line modifies the calculated spread to obtain an "unbiased spread estimation." It involves scaling the
            # initial spread by a factor based on the number of "good" points in the cluster (self.cluster.point_num). The formula
            # used is typically derived from statistical considerations to provide an unbiased estimate of the population
            # variance. (self.cluster.point_num + 1) / (self.cluster.point_num - 1) is a common correction factor used in statistics.
            # This factor adjusts the spread to account for the fact that when estimating population variance from a sample,
            # the sample variance tends to be biased low. The formula aims to correct this bias.
            if self.cluster.point_num != 1:
                spread = (
                    spread * (self.cluster.point_num + 1) / (self.cluster.point_num - 1)
                )

            # Ensure the computed spread estimation is between 1x and 2x of configured limits
            spread = min(2 * const.EKF_SPREAD_LIM[m], spread)
            spread = max(const.EKF_SPREAD_LIM[m], spread)

            if spread > self.spread_est[m]:
                self.spread_est[m] = spread
            else:
                self.spread_est[m] = (1.0 - const.EKF_A_SPR) * self.spread_est[
                    m
                ] + const.EKF_A_SPR * spread

    def associate_pointcloud(self, pointcloud: np.array):
        self.cluster = PointCluster(pointcloud)
        self._estimate_point_num()
        self._estimate_measurement_spread()

    def get_Rm(self):
        rm = np.diag(((self.spread_est / 2) ** 2))
        # print(f"Rm dim:", rm.shape)
        return rm

    def get_Rc(self):
        return self.get_Rm() / self.cluster.point_num

    def update_state(self):
        self.state.inst.update(np.array(self.cluster.centroid), R=self.get_Rc())


class TrackBuffer:
    def __init__(self):
        self.tracks = []
        # TODO: create a structure that erases the inactive tracks or just keeps the active ones.

        # This field keeps track of the iterations that passed until we have valid measurements
        self.dt_multiplier = 1

    def has_active_tracks(self):
        if len(self.tracks) != 0:
            return True
        else:
            return False

    def _calc_dist_fun(self, full_set):
        dist_matrix = np.empty((full_set.shape[0], len(self.tracks)))

        # simple_approach = np.empty((full_set.shape[0]))
        simple_approach = np.full(full_set.shape[0], None, dtype=object)

        for j, track in enumerate(self.tracks):
            # Find group residual covariance matrix
            # NOTE: Add D
            H_i = np.dot(const.EKF_H, track.state.inst.x).flatten()

            # TODO: This is wrong. Fix it
            # C_g_i = np.dot(np.dot(H_i, track.state.inst.P), H_i.T) + track.get_Rm
            C_g_i = track.get_Rm()

            for i, point in enumerate(full_set):
                # Find innovation for each measurement
                y_ij = np.array(point) - H_i

                # Find distance function (d^2)
                # TODO: This is also wrong
                # dist_matrix[i][j] = np.dot(np.dot(y_ij.T, np.linalg.inv(C_g_i)), y_ij)
                dist_matrix[i][j] = np.dot(np.dot(y_ij.T, C_g_i), y_ij)

                # Perform G threshold check
                if dist_matrix[i][j] < const.EKF_G:
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

    def predict_all(self):
        for track in self.tracks:
            track.predict_state(self.dt_multiplier)

    def update_all(self):
        for track in self.tracks:
            track.update_state()

    def associate_points_to_tracks(self, full_set):
        unassigned = []
        clusters = [[] for _ in range(len(self.tracks))]
        simple_matrix = self._calc_dist_fun(full_set)

        for i, point in enumerate(full_set):
            if simple_matrix[i] is None:
                unassigned.append(point)
            else:
                clusters[simple_matrix[i]].append(point)

        # TODO: Check for minimum number of points before associating to a track

        for j, track in enumerate(self.tracks):
            # TODO: Check if the cluster[j] is empty or it does not pass the threshold of minimum points
            # then increment the lifetime counter or perform state change or something.
            if len(clusters[j]) == 0:
                track.lifetime += 1
            else:
                track.associate_pointcloud(np.array(clusters[j]))

        return unassigned


def perform_tracking(pointcloud, trackbuffer: TrackBuffer):
    # Prediction Step
    trackbuffer.predict_all()

    # Association Step
    unassigned = trackbuffer.associate_points_to_tracks(pointcloud)
    # TODO: update tracks status or deem inactive.

    # Update Step
    trackbuffer.update_all()

    new_clusters = []
    # Clustering of the remainder step
    if len(unassigned) != 0:
        new_clusters = apply_DBscan(unassigned)

    # Create new track for every new cluster
    trackbuffer.add_tracks(new_clusters)
