import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from preprocess import preprocess_single_frame

import matplotlib.pyplot as plt
import numpy as np


class PostureEstimation:
    def __init__(self, model_dir):
        self.model = load_model(model_dir)

    def estimate_posture(self, frame_matrices):
        return self.model.predict(frame_matrices)
