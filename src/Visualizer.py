import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch

import constants as const
from Tracking import TrackBuffer, ClusterTrack
from Utils import calc_projection_points


def calc_fade_square(track: ClusterTrack):
    coords = np.array(
        [
            track.state.inst.x[0],
            track.state.inst.x[1],
            track.state.inst.x[2],
        ]
    ).flatten()
    center_x = calc_projection_points(value=coords[0], y=coords[1])
    center_z = track.height_buffer.get_mean()
    center = [center_x, center_z]
    rect_size = max(
        const.V_SCREEN_FADE_SIZE_MIN,
        track.width_buffer.get_max(),
        min(
            const.V_SCREEN_FADE_SIZE_MAX,
            const.V_SCREEN_FADE_SIZE_MAX - coords[1] * const.V_SCREEN_FADE_WEIGHT,
        ),
    )
    return (center, rect_size)


class Visualizer:
    def __init__(self):
        self.dynamic_art = []
        fig = plt.figure()
        axis_3d = const.V_3D_AXIS

        # Create subplot of raw pointcloud
        self.ax_raw = fig.add_subplot(121, projection="3d")
        self.scatter_raw = self.ax_raw.scatter([], [], [])
        self.ax_raw.set_xlim(-axis_3d[0] / 2, axis_3d[0] / 2)
        self.ax_raw.set_ylim(0, axis_3d[1])
        self.ax_raw.set_zlim(0, axis_3d[2])
        self.ax_raw.set_title("Scatter plot of raw Point Cloud")

        # Create subplot of tracks and predictions
        self.ax = fig.add_subplot(122, projection="3d")
        self.scatter = self.ax.scatter([], [], [])
        self.ax.set_xlim(-axis_3d[0] / 2, axis_3d[0] / 2)
        self.ax.set_ylim(-1, axis_3d[1])
        self.ax.set_zlim(0, axis_3d[2])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.invert_yaxis()
        self.ax.invert_xaxis()
        legend_handles = [
            Patch(color="red", label="Predicted Track"),
            Patch(color="green", label="Measured Track"),
        ]
        self.ax.legend(handles=legend_handles)

        plt.show(block=False)

    def clear(self):
        # Remove pointcloud
        if self.scatter is not None:
            self.scatter.remove()

        # Remove bounding boxes
        for collection in self.dynamic_art:
            collection.remove()
        self.dynamic_art = []

        # Remove screen fading
        for patch in self.ax.patches:
            patch.remove()

    def update_raw(self, x, y, z):
        # Update the data in the 3D scatter plot
        self.scatter_raw._offsets3d = (x, y, z)
        plt.draw()

    def _draw_bounding_box(self, x, color="gray", fill=0):
        # Create Bounding Boxes
        c = np.array(
            [
                x[0],
                x[1],
                x[2] * 0.0,
            ]
        ).flatten()
        vertices = np.array(
            [
                [-0.3, -0.3, 0],
                [0.3, -0.3, 0],
                [0.3, 0.3, 0],
                [-0.3, 0.3, 0],
                [-0.3, -0.3, 1.8],
                [0.3, -0.3, 1.8],
                [0.3, 0.3, 1.8],
                [-0.3, 0.3, 1.8],
            ]
        )
        vertices = vertices + c
        # vertices = vertices * (const.V_BBOX_HEIGHT / 6) + c
        # Define the cube faces
        faces = [
            [vertices[j] for j in [0, 1, 2, 3]],
            [vertices[j] for j in [4, 5, 6, 7]],
            [vertices[j] for j in [0, 3, 7, 4]],
            [vertices[j] for j in [1, 2, 6, 5]],
            [vertices[j] for j in [0, 1, 5, 4]],
            [vertices[j] for j in [2, 3, 7, 6]],
        ]

        cube = Poly3DCollection(faces, color=[color], alpha=fill)
        self.ax.add_collection3d(cube)
        return cube

    def draw_fading_window(self, track):
        (center, rect_size) = calc_fade_square(track)
        vertices = [
            (center[0] - rect_size / 2, 0, center[1] - rect_size / 2),
            (center[0] + rect_size / 2, 0, center[1] - rect_size / 2),
            (center[0] + rect_size / 2, 0, center[1] + rect_size / 2),
            (center[0] - rect_size / 2, 0, center[1] + rect_size / 2),
        ]

        # Create a Poly3DCollection
        rectangle = Poly3DCollection([vertices], facecolors="black", alpha=0.8)
        self.ax.add_collection3d(rectangle)
        return rectangle

    def update(self, trackbuffer: TrackBuffer):
        x_all = np.array([])  # Initialize as empty NumPy arrays
        y_all = np.array([])
        z_all = np.array([])
        color_all = np.array([]).reshape(0, 3)

        for track in trackbuffer.effective_tracks:
            # We want to visualize only new points.
            # if track.lifetime == 0:
            # coords = track.cluster.pointcloud
            # x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

            # # Update pointclouds with different colors for different clusters
            # x_all = np.concatenate([x_all, x])
            # y_all = np.concatenate([y_all, y])
            # z_all = np.concatenate([z_all, z])
            # color_all = np.concatenate(
            #     [color_all, np.repeat([track.color], len(x), axis=0)]
            # )

            self.dynamic_art.append(
                self._draw_bounding_box(track.state.inst.x, color=track.color, fill=0.5)
            )
            self.dynamic_art.append(
                self._draw_bounding_box(track.predict_x, color="red")
            )
            self.dynamic_art.append(
                self._draw_bounding_box(track.cluster.centroid, color="green")
            )
            # self.dynamic_art.append(self.draw_fading_window(track))

        # Update 3d plot
        self.scatter = self.ax.scatter(x_all, y_all, z_all, c=color_all, marker="o")
        self.ax.set_title(
            f"Track Number: {len(trackbuffer.effective_tracks)}", loc="left"
        )

    def draw(self):
        plt.draw()
        plt.pause(0.001)  # Pause for a short time to allow for updating


class ScreenAdapter:
    def __init__(self):
        self.win = pg.GraphicsLayoutWidget()
        self.view = self.win.addPlot()
        self.view.setAspectLocked()
        self.view.getViewBox().setBackgroundColor((255, 255, 255))
        self.view.setRange(
            xRange=(-const.V_3D_AXIS[0] / 2, const.V_3D_AXIS[0] / 2),
            yRange=(const.M_HEIGHT, const.V_3D_AXIS[2]),
        )
        self.view.invertX()
        self.win.showMaximized()

        # Create a scatter plot with squares
        brush = pg.mkBrush(color=(0, 0, 0))
        self.scatter = pg.ScatterPlotItem(pen=None, brush=brush, symbol="s")
        self.view.addItem(self.scatter)

        self.PIX_TO_M = 3779 * const.V_SCALLING

    def update(self, trackbuffer: TrackBuffer):
        # Clear previous items in the view
        self.scatter.clear()
        for track in trackbuffer.effective_tracks:
            center, rect_size = calc_fade_square(track)

            self.scatter.addPoints(
                x=[center[0] - rect_size / 2],
                y=[center[1] - rect_size / 2],
                size=rect_size * self.PIX_TO_M,
            )

        # Update the view
        QApplication.processEvents()
