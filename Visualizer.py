import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Visualizer:
    def __init__(
        self,
        enable_2d=False,
        axis_3d: [float, float, float] = [2.0, 2.0, 2.0],
        axis_2d: [float, float] = [1.5, 1.5],
        rect_size: float = 0.5,
    ):
        self.axis_3d = axis_3d
        self.axis_2d = axis_2d
        self.rect_size = rect_size

        fig = plt.figure()
        self.ax = fig.add_subplot(121, projection="3d")
        self.scatter = self.ax.scatter([], [], [])
        self.ax.set_xlim(0, axis_3d[0])
        self.ax.set_ylim(0, axis_3d[1])
        self.ax.set_zlim(0, axis_3d[2])

        self.rect_2d = enable_2d

        if self.rect_2d:
            # Create a 2D plot for the square
            self.ax2 = fig.add_subplot(122)
            self.ax2.set_xlim(-self.axis_2d[0], self.axis_2d[0])
            self.ax2.set_ylim(-self.axis_2d[1], self.axis_2d[1])

        plt.show(block=False)  # Set block=False to allow continuing execution

    def update(self, x, y, z):
        # Update the data in the 3D scatter plot
        self.scatter._offsets3d = (x, y, z)

        # Update the square in the 2D plot
        if self.rect_2d:
            for patch in self.ax2.patches:
                patch.remove()

            center = (-np.mean(x), np.mean(z))
            # Plot the filled square with updated center coordinates and alpha
            square = Rectangle(
                (center[0] - self.rect_size / 2, center[1] - self.rect_size / 2),
                self.rect_size,
                self.rect_size,
                alpha=0.5,
                color="b",
            )
            self.ax2.add_patch(square)

        plt.draw()
        plt.pause(0.1)  # Pause for a short time to allow for updating
