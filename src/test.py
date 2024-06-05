import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model
from sklearn import metrics


def single_case_joints():
    # Assuming your array of 19 floats is named 'data'
    num_full = np.load("./model/Accuracy/MARS_accuracy.npy")
    # print(np.mean(num[:19, 6]))

    # skip = [1, 3, 5, 7]
    # mask = np.ones(8, dtype=bool)
    # mask[skip] = False

    num_all = num_full[:19, 6]
    num_x = num_full[:19, 0]
    num_y = num_full[:19, 2]
    num_z = num_full[:19, 4]

    # num = num_full[:19, mask]

    # Generate x values (assuming index as x values)
    keypoint_ids = range(0, 19)

    # Plot the data
    plt.plot(
        keypoint_ids, num_x, marker="o", linestyle="-", label="x-axis", linewidth=0.75
    )
    plt.plot(
        keypoint_ids, num_y, marker="o", linestyle="-", label="z-axis", linewidth=0.75
    )
    plt.plot(
        keypoint_ids, num_z, marker="o", linestyle="-", label="y-axis", linewidth=0.75
    )
    plt.plot(
        keypoint_ids, num_all, marker="o", linestyle="-", label="Average", linewidth=3
    )
    # plt.plot(keypoint_ids, num, marker="o", linestyle="-")

    # Add labels and title
    plt.xlabel("Keypoint Index")
    plt.ylabel("MAE (cm)")
    plt.title("MAE for every keypoint")
    plt.xticks(keypoint_ids)  # Show grid
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()


def multi_cases_joints():
    # Assuming your array of 19 floats is named 'data'
    num_1 = np.load("../trained_cases/Rand_Base_Abs/model/Accuracy/MARS_accuracy.npy")
    num_2 = np.load("../trained_cases/Rand_Ours_Abs/model/Accuracy/MARS_accuracy.npy")
    num_3 = np.load("../trained_cases/Rand_Base_Rel/model/Accuracy/MARS_accuracy.npy")
    num_4 = np.load("../trained_cases/Rand_Base_Rel2/model/Accuracy/MARS_accuracy.npy")
    num_5 = np.load(
        "../trained_cases/Rand_Ours_Abs_Noise/model/Accuracy/MARS_accuracy.npy"
    )

    # print(np.mean(num[:19, 6]))

    # skip = [1, 3, 5, 7]
    # mask = np.ones(8, dtype=bool)
    # mask[skip] = False

    num_all = num_1[:19, 6]
    num_all2 = num_2[:19, 6]
    num_all3 = num_3[:19, 6]
    num_all4 = num_4[:19, 6]
    num_all5 = num_5[:19, 6]
    num_all6 = np.load("./full.npy")
    num_all6 *= 100

    print(np.mean(num_all2))

    # num = num_full[:19, mask]

    # Generate x values (assuming index as x values)
    keypoint_ids = range(0, 19)

    # Plot the data
    # plt.plot(
    #     keypoint_ids,
    #     num_all4,
    #     marker="o",
    #     linestyle="-",
    #     label="Baseline",
    #     linewidth=0.75,
    # )
    # plt.plot(
    #     keypoint_ids,
    #     num_all,
    #     marker="o",
    #     linestyle="-",
    #     label="Improvement 1",
    #     linewidth=0.75,
    # )
    plt.plot(
        keypoint_ids,
        num_all2,
        marker="o",
        linestyle="-",
        label="Dataset w/o noise",
        linewidth=0.75,
    )
    plt.plot(
        keypoint_ids,
        num_all3,
        marker="o",
        linestyle="-",
        label="Bottom-up Baseline",
        linewidth=0.75,
    )

    # plt.plot(
    #     keypoint_ids,
    #     num_all5,
    #     marker="o",
    #     linestyle="-",
    #     label="Dataset w/ noise",
    #     linewidth=0.75,
    # )
    plt.plot(
        keypoint_ids,
        num_all6,
        marker="o",
        linestyle="-",
        label="Top-down Improvement",
        linewidth=0.75,
    )
    # plt.plot(keypoint_ids, num, marker="o", linestyle="-")

    # Add labels and title
    plt.xlabel("Keypoint Index")
    plt.ylabel("Mean Average Error (cm)")
    plt.xticks(keypoint_ids)  # Show grid
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()


def multi_cases_average():
    num_1 = np.load("../trained_cases/Rand_Base_Abs/model/Accuracy/MARS_accuracy.npy")
    num_2 = np.load("../trained_cases/Rand_Ours_Abs/model/Accuracy/MARS_accuracy.npy")
    num_all_mae = [num_1[19, 6], num_2[19, 6]]
    num_all_rmse = [num_1[19, 7], num_2[19, 7]]

    temp_1 = np.load("../trained_cases/Rand_Base_Abs/model/Accuracy/all.npy")
    temp_2 = np.load("../trained_cases/Rand_Ours_Abs/model/Accuracy/all.npy")
    min1 = min2 = 10
    max1 = max2 = 0
    for i in range(10):
        mean1 = np.mean([temp_1[i][19][0], temp_1[i][19][2], temp_1[i][19][4]])
        if mean1 < min1:
            min1 = mean1
        if mean1 > max1:
            max1 = mean1

        mean2 = np.mean([temp_2[i][19][0], temp_2[i][19][2], temp_2[i][19][4]])
        if mean2 < min2:
            min2 = mean2
        if mean2 > max2:
            max2 = mean2

    asymmetric_error1 = [num_all_mae[0] - min1, max1 - num_all_mae[0]]
    asymmetric_error2 = [num_all_mae[1] - min2, max2 - num_all_mae[1]]

    print(num_all_mae)
    print(asymmetric_error1, asymmetric_error2)

    print(min1, max1)
    print(min2, max2)

    number_of_cases = [1, 2]

    # Plot the data
    # plt.plot(
    #     number_of_cases,
    #     num_all_mae,
    #     marker="o",
    #     linestyle="-",
    #     label="Baseline",
    #     linewidth=0.75,
    # )
    plt.bar(number_of_cases, num_all_rmse, yerr=asymmetric_error1, capsize=5, label="a")
    plt.bar(number_of_cases, num_all_mae, yerr=asymmetric_error2, capsize=5, label="a")

    # plt.plot(
    #     number_of_cases,
    #     num_all_rmse,
    #     marker="o",
    #     linestyle="-",
    #     label="Frame Fusion",
    #     linewidth=0.75,
    # )
    plt.xlabel("Keypoint Index")
    plt.ylabel("MAE (cm)")
    plt.title("MAE for every keypoint")
    plt.xticks(number_of_cases)  # Show grid
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

    # # Create a bar plot with error bars
    # plt.bar(x, y, yerr=error, capsize=5)


def line_plot_w_error_bars():
    # Example data
    x = np.arange(10)
    y = np.sin(x / 20 * np.pi)
    error = 0.1 + 0.1 * np.sqrt(x)  # Example error values

    # Create a line plot with error bars
    plt.errorbar(x, y, yerr=error, fmt="-o", capsize=5)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Line Plot with Error Bars")
    plt.grid(True)
    plt.show()


# COMPARISON: FULL BASELINE VS ABSOLUTE BASELINE
# I NEED THE GROUND TRUTH ABSOLUTE
# ALSO I NEED TO SAVE THE CLUSTER CENTROID


def absolute_compare():
    sets = [
        [
            "B11",
            "B12",
            "B13",
            "B14",
            "B15",
            "B41",
            "B42",
            "B43",
            "B44",
            "B45",
            "A51",
            "A52",
            "A53",
            "A54",
            "A55",
        ],
        [
            "B21",
            "B22",
            "B23",
            "B24",
            "B25",
            "B71",
            "B72",
            "B73",
            "B74",
            "B75",
            "B81",
            "B84",
            "B85",
        ],
        [
            "B21",
            "B22",
            "B23",
            "B24",
            "B25",
            "A31",
            "A32",
            "A33",
            "A34",
            "A35",
            "B51",
            "B52",
            "B53",
            "B54",
            "B55",
        ],
        [
            "B21",
            "B22",
            "B23",
            "B24",
            "B25",
            "A51",
            "A52",
            "A53",
            "A54",
            "A55",
            "B61",
            "B62",
            "B63",
            "B64",
            "B65",
        ],
        [
            "A21",
            "A22",
            "A23",
            "A24",
            "A25",
            "A31",
            "A32",
            "A33",
            "A34",
            "A35",
            "B41",
            "B42",
            "B43",
            "B44",
            "B45",
        ],
        [
            "B31",
            "B32",
            "B33",
            "B34",
            "B35",
            "A61",
            "A62",
            "A63",
            "A64",
            "A65",
            "A71",
            "A72",
            "A73",
            "A74",
            "A75",
        ],
        [
            "B31",
            "B32",
            "B33",
            "B34",
            "B35",
            "A41",
            "B41",
            "A42",
            "B42",
            "A43",
            "B43",
            "A44",
            "B44",
            "A45",
            "B45",
        ],
        [
            "A31",
            "A32",
            "A33",
            "A34",
            "A35",
            "A51",
            "A52",
            "A53",
            "A54",
            "A55",
            "B61",
            "B62",
            "B63",
            "B64",
            "B65",
        ],
        [
            "A61",
            "B61",
            "A62",
            "B62",
            "A63",
            "B63",
            "A64",
            "B64",
            "A65",
            "B65",
            "A71",
            "A72",
            "A73",
            "A74",
            "A75",
        ],
        [
            "B11",
            "B12",
            "B13",
            "B14",
            "B15",
            "B21",
            "B22",
            "B23",
            "B24",
            "B25",
            "B51",
            "B52",
            "B53",
            "B54",
            "B55",
        ],
    ]

    results = []
    for i in range(10):
        model = load_model(f"../trained_cases/model/{i}MARS.h5")
        testing_data = np.load(f"./dataset/formatted/mmWave/{i}/testing_mmWave.npy")
        ground_truth = np.load(f"./dataset/formatted/kinect/{i}/testing_labels.npy")
        predictions = model.predict(testing_data)
        centroids = concatenate_centroids(sets[i])

        predictions[:, :19] -= centroids[:, 0][:, np.newaxis]
        predictions[:, 38:] += centroids[:, 1][:, np.newaxis]

        x_mae = metrics.mean_absolute_error(
            ground_truth[:, 0:19], predictions[:, 0:19], multioutput="raw_values"
        )
        y_mae = metrics.mean_absolute_error(
            ground_truth[:, 19:38], predictions[:, 19:38], multioutput="raw_values"
        )
        z_mae = metrics.mean_absolute_error(
            ground_truth[:, 38:57], predictions[:, 38:57], multioutput="raw_values"
        )
        all_19_points_mae = np.concatenate((x_mae, y_mae, z_mae)).reshape(3, 19)
        avg_19_points_mae = np.mean(all_19_points_mae, axis=0)

        results.append(avg_19_points_mae)

    final_array = np.mean(np.array(results), axis=0)
    np.save("full.npy", final_array)


def concatenate_centroids(prefixes):
    concatenated_arrays = []
    folder_path = "./centroids/"

    # Iterate over each sorted prefix
    for prefix in prefixes:
        # Get a sorted list of filenames that start with the specified prefix
        sorted_filenames = sorted(
            filename
            for filename in os.listdir(folder_path)
            if filename.startswith(prefix)
        )

        # Iterate over sorted filenames
        for filename in sorted_filenames:
            # Load the array from the file
            array = np.load(os.path.join(folder_path, filename))
            # Append the array to the list of concatenated arrays
            concatenated_arrays.append(array)

    # Concatenate the arrays along the first axis (assuming arrays are of shape (N, 2))
    if concatenated_arrays:
        concatenated_result = np.concatenate(concatenated_arrays, axis=0)
        return concatenated_result
    else:
        return None  # Return None if no files were found


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
    sets = [
        [
            "B11",
            "B12",
            "B13",
            "B14",
            "B15",
            "B41",
            "B42",
            "B43",
            "B44",
            "B45",
            "A51",
            "A52",
            "A53",
            "A54",
            "A55",
        ],
        [
            "B21",
            "B22",
            "B23",
            "B24",
            "B25",
            "B71",
            "B72",
            "B73",
            "B74",
            "B75",
            "B81",
            "B84",
            "B85",
        ],
        [
            "B21",
            "B22",
            "B23",
            "B24",
            "B25",
            "A31",
            "A32",
            "A33",
            "A34",
            "A35",
            "B51",
            "B52",
            "B53",
            "B54",
            "B55",
        ],
        [
            "B21",
            "B22",
            "B23",
            "B24",
            "B25",
            "A51",
            "A52",
            "A53",
            "A54",
            "A55",
            "B61",
            "B62",
            "B63",
            "B64",
            "B65",
        ],
        [
            "A21",
            "A22",
            "A23",
            "A24",
            "A25",
            "A31",
            "A32",
            "A33",
            "A34",
            "A35",
            "B41",
            "B42",
            "B43",
            "B44",
            "B45",
        ],
        [
            "B31",
            "B32",
            "B33",
            "B34",
            "B35",
            "A61",
            "A62",
            "A63",
            "A64",
            "A65",
            "A71",
            "A72",
            "A73",
            "A74",
            "A75",
        ],
        [
            "B31",
            "B32",
            "B33",
            "B34",
            "B35",
            "A41",
            "B41",
            "A42",
            "B42",
            "A43",
            "B43",
            "A44",
            "B44",
            "A45",
            "B45",
        ],
        [
            "A31",
            "A32",
            "A33",
            "A34",
            "A35",
            "A51",
            "A52",
            "A53",
            "A54",
            "A55",
            "B61",
            "B62",
            "B63",
            "B64",
            "B65",
        ],
        [
            "A61",
            "B61",
            "A62",
            "B62",
            "A63",
            "B63",
            "A64",
            "B64",
            "A65",
            "B65",
            "A71",
            "A72",
            "A73",
            "A74",
            "A75",
        ],
        [
            "B11",
            "B12",
            "B13",
            "B14",
            "B15",
            "B21",
            "B22",
            "B23",
            "B24",
            "B25",
            "B51",
            "B52",
            "B53",
            "B54",
            "B55",
        ],
    ]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Load MARS model
    model = load_model("../trained_cases/model/MARS.h5")
    featuremap_test = np.load(
        "../mmWave_MSc/dataset/formatted/mmWave/0/testing_mmWave.npy"
    )
    ground_truth = np.load(
        "../mmWave_MSc/dataset/formatted/kinect/0/testing_labels.npy"
    )

    predictions = model.predict(featuremap_test)
    centroids = concatenate_centroids(sets[0])

    predictions[:, :19] -= centroids[:, 0][:, np.newaxis]
    predictions[:, 38:] += centroids[:, 1][:, np.newaxis]

    for predict_num in range(0, len(predictions)):

        ax.clear()  # Clear the plot before each iteration

        # NOTE: MARS outputs the keypoint coords as [x1, x2, ..., xN, y1, y2, ..., yN, z1, z2, ..., zN]
        reshaped_data = predictions[predict_num].reshape(3, -1)
        generate_skeleton(reshaped_data, ax)

        # GROUND TRUTH
        # print("%.10f" % ground_truth[predict_num])
        reshaped_ground_truth = ground_truth[predict_num].reshape(3, -1)
        generate_skeleton(reshaped_ground_truth, ax, True)

        # Set fixed axis scales
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(0, 3)

        if save:
            plt.savefig(f"./gif/{predict_num}.png")
        else:
            plt.draw()
            plt.pause(0.001)

    plt.show()  # Move plt.show() outside of the loop to display the final plot


def motion_model_eval():
    pass


# multi_cases_joints()
# line_plot_w_error_bars()
# multi_cases_average()
# absolute_compare()
# model_demo()

# all_a = np.load("../trained_cases/Rand_Base_Abs/model/Accuracy/all.npy")
# all_a2 = np.around(all_a[2], decimals=2)

# all = np.load("./dataset/formatted/mmWave/0/testing_mmWave.npy")

# print(all_a.shape)


temp = np.array(
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
    ]
)

temp1 = temp.reshape(3, -1)
print(temp1)
