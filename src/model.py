import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from preprocess import preprocess_single_frame

# Load MARS model
model = load_model("../MARS/model/MARS.h5")

# data = preprocess_single_frame(raw_array)
data = []
featuremap_test = np.load("../MARS/feature/featuremap_test.npy")

# data.append(featuremap_test[0])
# For example, if your test data is stored in a numpy array called test_data.npy
# test_data = np.load(data)

# Make predictions on the test data
predictions = model.predict(featuremap_test)

print(predictions)

reshaped_data = predictions[2].reshape(-1, 3)  # Reshape to have 3 columns

x_all = np.array(reshaped_data[:, 0])  # Initialize as empty NumPy arrays
y_all = np.array(reshaped_data[:, 1])
z_all = np.array(reshaped_data[:, 2])

connections = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (0, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
]

fig = plt.figure()
# Create subplot of raw pointcloud
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_title("Skeletal reconstruction")
scatter = ax.scatter(x_all, y_all, z_all, c="black", marker="o")
scatter = ax.scatter(x_all[3], y_all[3], z_all[3], c="red", marker="o")

for connection in connections:
    ax.plot(
        [x_all[connection[0]], x_all[connection[1]]],
        [y_all[connection[0]], y_all[connection[1]]],
        [z_all[connection[0]], z_all[connection[1]]],
        color="blue",
    )

plt.draw()
plt.show()
# print(predictions)
