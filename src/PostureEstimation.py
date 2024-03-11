import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from preprocess import preprocess_single_frame

import matplotlib.pyplot as plt
import numpy as np


class PostureEstimation:
    def __init__(self, model_dir):
        self.model = load_model(model_dir)

    def estimate_posture(self, pointcloud):
        # NOTE: The input is in the form of [x, y, z, x', y', z', r', s]
        fixed_pointcloud = pointcloud[:, [0, 1, 2, -2, -1]]
        feature_matrix = preprocess_single_frame(fixed_pointcloud)

        predictions = self.model.predict(feature_matrix)

        return predictions
