import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
from preprocess import preprocess_single_frame

# Define connections and keypoints
connections = [
    (0, 1),  # SpineBase to SpineMid
    (1, 2),  # SpineMid to Neck
    (2, 3),  # Neck to Head
    (2, 4),  # Neck to ShoulderLeft
    (2, 7),  # Neck to ShoulderRight
    (4, 5),  # ShoulderLeft to ElbowLeft
    (5, 6),  # ElbowLeft to WristLeft
    (7, 8),  # ShoulderRight to ElbowRight
    (8, 9),  # ElbowRight to WristRight
    (0, 14),  # SpineBase to HipRight
    (14, 15),  # HipRight to KneeRight
    (15, 16),  # KneeRight to AnkleRight
    (16, 17),  # AnkleRight to FootRight
    (0, 10),  # SpineBase to HipLeft
    (10, 11),  # HipLeft to KneeLeft
    (11, 12),  # KneeLeft to AnkleLeft
    (12, 13),  # AnkleLeft to FootLeft
    (2, 18),  # Neck to SpineShoulder
]

# Define keypoint colors
keypoint_colors = {
    0: "blue",  # SpineBase,
    1: "blue",  # SpineMid,
    2: "blue",  # Neck,
    3: "red",  # Head,
    4: "blue",  # ShoulderLeft,
    5: "green",  # ElbowLeft,
    6: "green",  # WristLeft,
    7: "blue",  # ShoulderRight,
    8: "green",  # ElbowRight,
    9: "green",  # WristRight,
    10: "blue",  # HipLeft,
    11: "green",  # KneeLeft,
    12: "green",  # AnkleLeft,
    13: "green",  # FootLeft,
    14: "blue",  # HipRight,
    15: "green",  # KneeRight,
    16: "green",  # AnkleRight,
    17: "green",  # FootRight,
    18: "blue",  # SpineShoulder
}
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-4, 4)
ax.set_title("Skeletal reconstruction")

# Load MARS model
model = load_model("../MARS/model/MARS.h5")

# data = preprocess_single_frame(raw_array)
data = []
featuremap_test = np.load("../MARS/feature/featuremap_test.npy")
# featuremap_test = np.load("../MARS/feature/labels_test.npy")

predictions = model.predict(featuremap_test)

for i in range(200):
    ax.clear()  # Clear the plot before each iteration

    reshaped_data = predictions[i].reshape(3, -1)

    for connection in connections:
        keypoint_1 = connection[0]
        keypoint_2 = connection[1]

        x_values = [reshaped_data[0][keypoint_1], reshaped_data[0][keypoint_2]]
        y_values = [reshaped_data[1][keypoint_1], reshaped_data[1][keypoint_2]]
        z_values = [reshaped_data[2][keypoint_1], reshaped_data[2][keypoint_2]]

        ax.plot(x_values, y_values, z_values, color="black")

    for keypoint_index in range(len(reshaped_data[0])):
        color = keypoint_colors.get(keypoint_index, "blue")
        marker = "o" if keypoint_index != 3 else "s"  # Use square marker for the head
        ax.scatter(
            reshaped_data[0][keypoint_index],
            reshaped_data[1][keypoint_index],
            reshaped_data[2][keypoint_index],
            c=color,
            marker=marker,
            s=100 if keypoint_index == 3 else 50,  # Larger size for the head
        )

    # Set fixed axis scales
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    plt.draw()
    plt.pause(0.1)

plt.show()  # Move plt.show() outside of the loop to display the final plot
