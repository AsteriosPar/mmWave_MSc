import torch
import numpy as np
import pandas as pd
import os
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import PointNetConv
import constants as const


class PointCloudData:
    def __init__(self, dataset_path):
        # Get labels by retrieving the names of the directories under the dataset_path
        self.directories = []

        for root, dirs, _ in os.walk(dataset_path):
            # Every directory name represents the classification label.
            for label in dirs:
                label_dir = os.path.join(dataset_path, label)
                # Every file represents a different pointcloud
                for datacloud in os.listdir(label_dir):
                    self.directories.append((os.path.join(label_dir, datacloud), label))

    def get_pointcloud(self, id):
        df = pd.read_csv(self.directories[id][0], header=None)
        points_coords = df.values.astype(float)

        return torch.Tensor(points_coords), self.directories[id][1]

    def __len__(self):
        return len(self.directories)


# pointset = PointCloudData(const.P_LOG_PATH)

# pointset.get_pointcloud(1)


# # Instantiate the PointNet model
# model = PointNetConv()

# # Example input (replace with your actual input data)
# batch_size = 32
# num_points = 1024
# num_features = 3  # Assuming XYZ coordinates

# # Create a dummy input tensor with shape (batch_size, num_points, num_features)
# input_data = torch.randn(batch_size, num_points, num_features)

# # Forward pass through the PointNet model
# output = model(input_data)
