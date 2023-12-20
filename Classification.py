import torch
import numpy as np
import json
import os


class PointCloudData:
    def __init__(self, dataset_path):
        # Get labels by retrieving the names of the directories under the dataset_path
        self.directories = []

        for root, dirs, _ in os.walk(dataset_path):
            for label in dirs:
                # Every directory name represents the classification label.
                directory_path = os.path.join(root, label)
                self.directories.append((directory_path, label))

        self.frameNumber = -1

    def get_pointcloud(self, id):
        with open(self.directories[id][0], "r") as file:
            datacloud = json.load(file)

            points_coords = [
                [point["X"], point["Y"], point["Z"]] for point in datacloud
            ]

        return torch.Tensor(np.array(points_coords)), torch.Tensor(
            [self.directories[id][0]]
        )
