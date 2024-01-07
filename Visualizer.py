import numpy as np
import math
import matplotlib.pyplot as plt
import constants as const
from constants import M_X, M_Y, M_Z
from matplotlib.patches import Rectangle
from matplotlib import gridspec
from Tracking import TrackBuffer


class Visualizer:
    def __init__(
        self,
        enable_2d=const.ENABLE_2D_VIEW,
        enable_cluster=const.ENABLE_3D_VIEW,
        axis_3d: [float, float, float] = [2.0, 4.0, 2.0],
        axis_2d: [float, float] = [1.5, 1.5],
        rect_size: float = 0.5,
    ):
        self.axis_3d = axis_3d
        self.axis_2d = axis_2d
        self.rect_size = rect_size

        fig = plt.figure()

        # Use gridspec to create a 1x3 grid for subplots
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        self.ax = fig.add_subplot(gs[0, 0], projection="3d")
        self.scatter = self.ax.scatter([], [], [])
        self.ax.set_xlim(-axis_3d[0] / 2, axis_3d[0] / 2)
        self.ax.set_ylim(0, axis_3d[1])
        self.ax.set_zlim(0, axis_3d[2])
        self.ax.set_title("Scatter plot of raw Point Cloud")

        self.rect_2d = enable_2d
        self.cluster = enable_cluster

        # if self.rect_2d:
        #     # Create a 2D plot for the square
        #     self.ax2 = fig.add_subplot(gs[0, 1])
        #     self.ax2.set_xlim(-self.axis_2d[0], self.axis_2d[0])
        #     self.ax2.set_ylim(-self.axis_2d[1], self.axis_2d[1])
        #     self.ax2.set_title("Vertical 2D square projection")

        if self.cluster:
            self.ax3 = fig.add_subplot(gs[1, 0], projection="3d")
            self.scatter3 = self.ax3.scatter([], [], [])
            self.ax3.set_xlim(-axis_3d[0] / 2, axis_3d[0] / 2)
            self.ax3.set_ylim(0, axis_3d[1])
            self.ax3.set_zlim(0, axis_3d[2])
            self.ax3.set_xlabel("X")
            self.ax3.set_ylabel("Y")
            self.ax3.set_zlabel("Z")

        plt.show(block=False)  # Set block=False to allow continuing execution

    def calc_square(self, x, y, z):
        y_dist = y - M_Y
        x_dist = x - M_X
        z_dist = z - M_Z

        x1 = -M_Y / (y_dist / x_dist)
        z1 = -M_Y / (y_dist / z_dist)

        screen_x = x1 + M_X
        screen_z = z1 + M_Z

        return (screen_x, screen_z)

    def update_raw(self, x, y, z):
        # Update the data in the 3D scatter plot
        self.scatter._offsets3d = (x, y, z)
        plt.draw()

    def update(self, trackbuffer: TrackBuffer):
        if len(trackbuffer.tracks) != 0:
            self.scatter3.remove()

        x_all = np.array([])  # Initialize as empty NumPy arrays
        y_all = np.array([])
        z_all = np.array([])
        color_all = np.array([]).reshape(0, 3)

        for track in trackbuffer.effective_tracks:
            # We want to visualize only new points.
            if track.lifetime == 0:
                coords = track.cluster.pointcloud
                x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

                # Concatenate arrays directly
                x_all = np.concatenate([x_all, x])
                y_all = np.concatenate([y_all, y])
                z_all = np.concatenate([z_all, z])
                color_all = np.concatenate(
                    [color_all, np.repeat([track.color], len(x), axis=0)]
                )

        # Update 3d plot
        self.scatter3 = self.ax3.scatter(x_all, y_all, z_all, c=color_all, marker="o")
        self.ax3.set_title(
            f"Track Number: {len(trackbuffer.effective_tracks)}", loc="left"
        )

        # Update the square in the 2D plot
        # if self.rect_2d:
        #     for patch in self.ax2.patches:
        #         patch.remove()

        #     xo, yo, zo = np.mean(x), np.mean(y), np.mean(z)
        #     center = self.calc_square(xo, yo, zo)
        #     # Plot the filled square with updated center coordinates and alpha
        #     square = Rectangle(
        #         (center[0] - self.rect_size / 2, center[1] - self.rect_size / 2),
        #         self.rect_size,
        #         self.rect_size,
        #         alpha=0.5,
        #         color="b",
        #     )
        #     self.ax2.add_patch(square)

        # TODO: Add bounding boxes

        plt.draw()
        plt.pause(0.1)  # Pause for a short time to allow for updating
