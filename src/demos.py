import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import constants as const
from PIL import Image
from keras.models import load_model


CONNECTIONS = [
    (3, 2),
    (2, 18),
    (18, 1),
    (1, 0),
    (5, 6),
    (18, 7),
    (7, 8),
    (8, 9),
    (18, 4),
    (4, 5),
    (0, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (0, 10),
    (10, 11),
    (11, 12),
    (12, 13),
]

CONNECTIONS_NPY = [
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
KEYPOINT_COLORS = [
    "blue",  # SpineBase,
    "blue",  # SpineMid,
    "blue",  # Neck,
    "red",  # Head,
    "blue",  # ShoulderLeft,
    "green",  # ElbowLeft,
    "green",  # WristLeft,
    "blue",  # ShoulderRight,
    "green",  # ElbowRight,
    "green",  # WristRight,
    "blue",  # HipLeft,
    "green",  # KneeLeft,
    "green",  # AnkleLeft,
    "green",  # FootLeft,
    "blue",  # HipRight,
    "green",  # KneeRight,
    "green",  # AnkleRight,
    "green",  # FootRight,
    "blue",  # SpineShoulder
]


def combined_plot(
    mmWave_data: pd.DataFrame, pointcloud_index, kinect_data, ax, save=False
):
    # Set up pointcloud points
    Xp = []
    Yp = []
    Zp = []
    frame_found = False
    for _, row in mmWave_data.iterrows():
        if row[0] == pointcloud_index:
            Xp.append(row[1])
            Yp.append(row[2])
            Zp.append(row[3])
            frame_found = True
        elif frame_found:
            break

    Xp = [-x for x in Xp]

    # Set up joint keypoints
    # NOTE: the mmWave microcontroller is placed 0.2m to the right of the Kinect sensor.
    x = [kinect_data.iloc[j] - 0.2 for j in range(2, 57, 3)]
    y = [kinect_data.iloc[j] for j in range(3, 58, 3)]
    z = [kinect_data.iloc[j] for j in range(4, 59, 3)]

    print(z[3], y[3])

    ax.clear()  # Clear the plot before each iteration
    ax.scatter(x, z, y)
    ax.scatter(Xp, Yp, Zp, s=10)
    for connection in CONNECTIONS:
        start_x, start_y, start_z = (
            x[connection[0]],
            y[connection[0]],
            z[connection[0]],
        )
        end_x, end_y, end_z = x[connection[1]], y[connection[1]], z[connection[1]]

        ax.plot([start_x, end_x], [start_z, end_z], [start_y, end_y], color="black")

    # Set fixed axis scales
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 4.5)
    ax.set_zlim(0, 3)

    if save:
        plt.savefig(f"./gif/{int(pointcloud_index)}.png")
    else:
        plt.draw()
        plt.pause(0.05)


def pair_demo(exp):
    labels = f"./dataset/log/kinect/{exp}.csv"
    pointclouds_path = f"./dataset/log/mmWave/{exp}"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    df2 = pd.read_csv(labels, header=None)

    filenames = os.listdir(pointclouds_path)
    filenames_sorted = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))
    for filename in filenames_sorted:
        with open(os.path.join(pointclouds_path, filename), "r") as file:
            df1 = pd.read_csv(file, header=None)
            unique_frames = df1.drop_duplicates(subset=0)

            for _, row1 in unique_frames.iterrows():
                timestamp1 = row1[6]
                closest_row = df2.iloc[(df2[0] - timestamp1).abs().argsort()[:1]]
                timestamp2 = closest_row.iloc[0, 0]

                if abs(timestamp1 - timestamp2) < 50:
                    combined_plot(df1, row1[0], closest_row.iloc[0, :], ax)
    plt.show()


def kinect_demo(labels="./dataset/log/kinect/time2.csv"):
    df = pd.read_csv(labels, header=None)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in range(len(df)):
        ax.clear()  # Clear the plot before each iteration

        x = [df.iloc[i, j] for j in range(2, 57, 3)]
        y = [df.iloc[i, j] for j in range(3, 58, 3)]
        z = [df.iloc[i, j] for j in range(4, 59, 3)]
        ax.scatter(x, z, y)

        for connection in CONNECTIONS:
            start_x, start_y, start_z = (
                x[connection[0]],
                y[connection[0]],
                z[connection[0]],
            )
            end_x, end_y, end_z = x[connection[1]], y[connection[1]], z[connection[1]]

            ax.plot([start_x, end_x], [start_z, end_z], [start_y, end_y], color="black")

        # Set fixed axis scales
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 3)
        ax.set_zlim(0, 3)

        plt.draw()
        plt.pause(0.1)
    plt.show()


def preprocessed_demo(experiment=const.P_EXPERIMENT_FILE_READ):
    kinect_input = os.path.join(
        f"{const.P_PREPROCESS_PATH}{const.P_KINECT_DIR}", f"{experiment}.csv"
    )
    mmwave_input = os.path.join(
        f"{const.P_PREPROCESS_PATH}{const.P_MMWAVE_DIR}", experiment
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    df2 = pd.read_csv(kinect_input, header=None)
    filenames = os.listdir(mmwave_input)
    filenames_sorted = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))
    kinect_index = 0
    for filename in filenames_sorted:
        with open(os.path.join(mmwave_input, filename), "r") as file:
            df1 = pd.read_csv(file, header=None)
            unique_frames = df1.drop_duplicates(subset=0)
            for _, row in unique_frames.iterrows():
                combined_plot(df1, row[0], df2.iloc[kinect_index, :], ax)
                kinect_index += 1

    plt.show()


def generate_skeleton(reshaped_data, ax, ground_truth=False):
    for connection in CONNECTIONS_NPY:
        x_values = [reshaped_data[0][connection[0]], reshaped_data[0][connection[1]]]
        y_values = [reshaped_data[1][connection[0]], reshaped_data[1][connection[1]]]
        z_values = [reshaped_data[2][connection[0]], reshaped_data[2][connection[1]]]

        ax.plot(x_values, z_values, y_values, color="black")

    for keypoint_index in range(len(reshaped_data[0])):
        if not ground_truth:
            color = KEYPOINT_COLORS[keypoint_index]
        else:
            color = "gray"

        ax.scatter(
            reshaped_data[0][keypoint_index],
            reshaped_data[2][keypoint_index],
            reshaped_data[1][keypoint_index],
            c=color,
            marker="o",
            s=100 if keypoint_index == 3 else 50,  # Larger size for the head
        )


def model_demo(save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Load MARS model
    model = load_model("./model/MARS.h5")
    featuremap_test = np.load(
        "../mmWave_MSc/dataset/formatted/mmWave/testing_mmWave.npy"
    )
    ground_truth = np.load("../mmWave_MSc/dataset/formatted/kinect/testing_labels.npy")

    predictions = model.predict(featuremap_test)

    for predict_num, prediction in enumerate(predictions):

        ax.clear()  # Clear the plot before each iteration

        # NOTE: MARS outputs the keypoint coords as [x1, x2, ..., xN, y1, y2, ..., yN, z1, z2, ..., zN]
        reshaped_data = prediction.reshape(3, -1)
        generate_skeleton(reshaped_data, ax)

        # GROUND TRUTH
        reshaped_ground_truth = ground_truth[predict_num].reshape(3, -1)
        generate_skeleton(reshaped_ground_truth, ax, True)

        # Set fixed axis scales
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 4)
        ax.set_zlim(-1, 3)

        if save:
            plt.savefig(f"./gif/{predict_num}.png")
        else:
            plt.draw()
            plt.pause(0.001)

    plt.show()  # Move plt.show() outside of the loop to display the final plot


def create_animation():
    images = [Image.open(f"./src/gif/{n}.png") for n in range(27, 226)]
    images[0].save(
        "ball.gif", save_all=True, append_images=images[1:], duration=120, loop=0
    )


# create_animation()
# pair_demo("A41")
# preprocessed_demo("testing/A41")
model_demo()
