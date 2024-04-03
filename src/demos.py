import matplotlib.pyplot as plt
import pandas as pd
import os
import os
import constants as const


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


def combined_plot(mmWave_data: pd.DataFrame, pointcloud_index, kinect_data, ax):
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

    ax.clear()  # Clear the plot before each iteration
    ax.scatter(x, z, y)
    ax.scatter(Xp, Yp, Zp)

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


preprocessed_demo("A43")
