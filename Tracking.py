import numpy as np
import math
import constants as const
from itertools import groupby
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag


def split_clusters(points, labels):
    sorted_data = sorted(zip(labels, points), key=lambda x: x[0])

    # Use groupby to group points by label
    grouped_points = {
        label: [p[1] for p in group]
        for label, group in groupby(sorted_data, key=lambda x: x[0])
    }

    # Convert the result to a list of lists (optional)
    result = [grouped_points[label] for label in sorted(set(labels))]

    return result


def get_centroid(cluster_coords):
    coords = np.array(cluster_coords)
    center = np.mean(coords, axis=0)
    return center


class PointCluster:
    def __init__(self, pointcloud, doppler):
        self.centroid = self.calc_centroid(pointcloud)
        self.rad_v = self.calc_doppler_mean(doppler)
        # Gets populated only after EKF
        self.v = None

    def calc_centroid(self, pointcloud):
        # Input pointcloud is a list of tuples(x,y,z)
        return np.mean(np.array(pointcloud), axis=0)

    def calc_doppler_mean(self, doppler):
        return np.mean(np.array(doppler))


class ClusterTrack:
    def __init__(self, cluster: PointCluster):
        self.id = 0
        self.cluster: PointCluster = cluster
        self.state_pred = None
        self.state = None
        self.status = 1

    # def calc_gate(self, input_points):
    # self._predict()

    # J = const.jacobian_matrix(self.state_pred[0])
    # H = const.measurent_matrix(self.state_pred[0])
    # R_m = 0
    # C_g = (
    #     np.dot(np.dot(J, self.state_pred[1]), J.T) + R_m
    # )  # group residual covariance matrix

    # for point in input_points:
    #     dist_fun = np.dot(
    #         np.dot(np.array(point - H).T, np.linalg.inv(C_g)), np.array(point - H)
    #     )

    # # Find the geometry of the gate according to the prediction output: self.state_pred

    # return gate

    # def update(self, ):
    #     self.


class TrackBuffer:
    def __init__(self):
        self.tracks: ClusterTrack = []
        self.lifetime: []

    def update(self):
        for track in self.tracks:
            if track.status != 0:
                track.update()

    def associate_point_to_track(self):
        for track in self.tracks:
            track.calc_gate()
            # For each point find the gate(s) it belongs
            # In case it is more than one assign it to the one according to the best distance metric
            # In case it is assigned to zero, add it to the unassigned list.


class Kalman:
    def __init__(self):
        # My state variables are: [x y z x' y' z']
        # My input variables are: [x y z]
        # No control function
        self.inst = KalmanFilter(dim_x=6, dim_z=3)

        self.inst.F = const.EKF_F
        self.inst.H = const.EKF_H

        # We assume independent noise in the x,y,z variables of equal standard deviations.
        self.inst.Q = const.EKF_Q_DISCR
        self.inst.R = np.eye(3) * const.EKF_R_STD**2

        # For initial values
        self.inst.x = np.array([[0, 0, 0, 0, 0, 0]]).T
        self.inst.P = np.eye(6) * 100.0

    def predict(self):
        self.inst.predict()
        return self.inst.x, self.inst.P
