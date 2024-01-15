import numpy as np
import matplotlib.pyplot as plt
import constants as const
from constants import M_X, M_Y, M_Z
from Tracking import TrackBuffer, ClusterTrack
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch


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

        # Create subplot of
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
        # Create proxy artists for the legend
        legend_handles = [
            Patch(color="red", label="Predicted Track"),
            Patch(color="green", label="Measured Track"),
        ]

        # Add legend
        self.ax.legend(handles=legend_handles)

        # Plot window surface
        rect_height = axis_3d[2]
        vertices = [
            (-axis_3d[0] / 2, 0, 0),
            (-axis_3d[0] / 2, 0, rect_height),
            (axis_3d[0] / 2, 0, rect_height),
            (axis_3d[0] / 2, 0, 0),
        ]
        rectangle = Poly3DCollection([vertices], facecolors="gray", alpha=0.3)
        self.ax.add_collection3d(rectangle)

        # Plot monitor position
        center = (const.M_X, const.M_Y, const.M_Z)
        radius = 0.1
        phi, theta = np.mgrid[0.0 : 2.0 * np.pi : 100j, 0.0 : np.pi : 50j]
        x_sphere = center[0] + radius * np.sin(theta) * np.cos(phi)
        y_sphere = center[1] + radius * np.sin(theta) * np.sin(phi)
        z_sphere = center[2] + radius * np.cos(theta)
        self.ax.plot_surface(x_sphere, y_sphere, z_sphere, color="red")

        plt.show(block=False)  # Set block=False to allow continuing execution

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

    def _calc_square(self, x, y, z):
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

    def _draw_screen_fade(self, track):
        coords = np.array(
            [
                track.state.inst.x[0],
                track.state.inst.x[1],
                track.state.inst.x[2],
            ]
        ).flatten()
        center = self._calc_square(coords[0], coords[1], coords[2])
        rect_size = const.V_SCREEN_FADE_SIZE
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
            if track.lifetime == 0:
                coords = track.cluster.pointcloud
                x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

                # Update pointclouds with different colors for different clusters
                x_all = np.concatenate([x_all, x])
                y_all = np.concatenate([y_all, y])
                z_all = np.concatenate([z_all, z])
                color_all = np.concatenate(
                    [color_all, np.repeat([track.color], len(x), axis=0)]
                )

                self.dynamic_art.append(
                    self._draw_bounding_box(
                        track.state.inst.x, color=track.color, fill=0.5
                    )
                )
                self.dynamic_art.append(
                    self._draw_bounding_box(track.predict_x, color="red")
                )
                self.dynamic_art.append(
                    self._draw_bounding_box(track.cluster.centroid, color="green")
                )
            self.dynamic_art.append(self._draw_screen_fade(track))

        # Update 3d plot
        self.scatter = self.ax.scatter(x_all, y_all, z_all, c=color_all, marker="o")
        self.ax.set_title(
            f"Track Number: {len(trackbuffer.effective_tracks)}", loc="left"
        )

    def draw(self):
        plt.draw()
        plt.pause(0.05)  # Pause for a short time to allow for updating


class ScreenAdapter:
    def __init__(self):
        self.monitor_coords = [const.M_X, const.M_Y, const.M_Z]
        self.sensor_coords = [0, 0, const.S_HEIGHT]
        self.rect_size = const.V_SCREEN_FADE_SIZE

    def _fade_center(self, x, y, z):
        y_dist = y - self.monitor_coords[1]
        x_dist = x - self.monitor_coords[0]
        z_dist = z - self.monitor_coords[2]

        x1 = -self.monitor_coords[1] / (y_dist / x_dist)
        z1 = -self.monitor_coords[1] / (y_dist / z_dist)

        screen_x = x1 + self.monitor_coords[0]
        screen_z = z1 + self.monitor_coords[2]

        return (screen_x, screen_z)

    def fade_shape(self, track: ClusterTrack):
        coords = np.array(
            [
                track.state.inst.x[0],
                track.state.inst.x[1],
                track.state.inst.x[2] + const.S_HEIGHT,
            ]
        ).flatten()
        center = self._fade_center(coords[0], coords[1], coords[2])
        vertices = [
            (center[0] - self.rect_size / 2, 0, center[1] - self.rect_size / 2),
            (center[0] + self.rect_size / 2, 0, center[1] - self.rect_size / 2),
            (center[0] + self.rect_size / 2, 0, center[1] + self.rect_size / 2),
            (center[0] - self.rect_size / 2, 0, center[1] + self.rect_size / 2),
        ]

        # Create a Poly3DCollection
        rectangle = Poly3DCollection([vertices], facecolors="black", alpha=1)
        return rectangle
