import os
import pandas as pd
import numpy as np
import constants as const
import csv


def preprocess_csv(experiment_name, threshold):
    input_path = os.path.join(const.P_LOG_PATH, experiment_name)
    output_path = os.path.join(const.P_PREPROCESS_PATH, experiment_name)

    for filename in os.listdir(input_path):
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.join(output_path, filename))
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(input_path, filename), "r") as file:
            # Read the CSV file without header
            df = pd.read_csv(file, header=None)

            # Randomly discard points within frames until the number of points per frame <= threshold
            df["num_points"] = df.groupby(0)[0].transform(
                "count"
            )  # Assuming frame_id is in the first column
            frames_to_reduce = df[0].unique()
            for frame_id in frames_to_reduce:
                frame_indices = df[df[0] == frame_id].index
                while len(frame_indices) > threshold:
                    index_to_drop = np.random.choice(frame_indices)
                    df.drop(index_to_drop, inplace=True)
                    frame_indices = df[df[0] == frame_id].index

            # Padding
            max_points = threshold
            num_points_per_frame = df.groupby(0)[
                0
            ].count()  # Number of points per frame after discarding
            for frame_id, num_points in num_points_per_frame.items():
                if num_points < max_points:
                    padding_rows = max_points - num_points
                    padding_data = [
                        [frame_id] + [0, 0, 0, 0] for _ in range(padding_rows)
                    ]  # Create padding data with zeros
                    padding = pd.DataFrame(
                        padding_data, columns=[0, 1, 2, 3, 4]
                    )  # Assuming x, y, z, velocity are in columns 1, 2, 3, 4 respectively
                    frame_index = df[
                        df[0] == frame_id
                    ].index.max()  # Find the index of the last row for the frame
                    df = pd.concat(
                        [
                            df.iloc[: frame_index + 1],
                            padding,
                            df.iloc[frame_index + 1 :],
                        ],
                        ignore_index=True,
                    )

            # Drop the 'num_points' column
            df.drop("num_points", axis=1, inplace=True)

            # Write to output CSV
            df.to_csv(
                os.path.join(output_path, filename), index=False, header=False
            )  # No header in the output CSV


def preprocess_npy(experiment_name, threshold):
    input_path = os.path.join(const.P_LOG_PATH, experiment_name)
    output_path = os.path.join(const.P_PREPROCESS_PATH, experiment_name)

    for filename in os.listdir(input_path):
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.join(output_path, filename))
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(input_path, filename), "r") as file:
            # Read the CSV file without header
            df = pd.read_csv(file, header=None)

            # Randomly discard points within frames until the number of points per frame <= threshold
            df["num_points"] = df.groupby(0)[0].transform(
                "count"
            )  # Assuming frame_id is in the first column
            frames_to_reduce = df[0].unique()
            for frame_id in frames_to_reduce:
                frame_indices = df[df[0] == frame_id].index
                while len(frame_indices) > threshold:
                    index_to_drop = np.random.choice(frame_indices)
                    df.drop(index_to_drop, inplace=True)
                    frame_indices = df[df[0] == frame_id].index

            # Padding
            max_points = threshold
            num_points_per_frame = df.groupby(0)[
                0
            ].count()  # Number of points per frame after discarding
            for frame_id, num_points in num_points_per_frame.items():
                if num_points < max_points:
                    padding_rows = max_points - num_points
                    padding_data = np.zeros(
                        (padding_rows, df.shape[1])
                    )  # Create padding data with zeros
                    padding_data[:, 0] = frame_id  # Set frame_id in the first column
                    df = pd.concat(
                        [
                            df[df[0] == frame_id],  # Keep existing data for the frame
                            pd.DataFrame(padding_data),  # Add padding data
                            df[df[0] != frame_id],  # Add remaining data
                        ],
                        ignore_index=True,
                    )

            # Drop the 'num_points' column
            df.drop("num_points", axis=1, inplace=True)

            # Convert DataFrame to NumPy array
            processed_data = df.to_numpy()

            # Save to output .npy file
            np.save(
                os.path.join(output_path, os.path.splitext(filename)[0] + ".npy"),
                processed_data,
            )


def find_intensity_normalizers(experiment_name="fixed"):
    intensities = []
    input_path = os.path.join(const.P_LOG_PATH, experiment_name)
    for filename in os.listdir(input_path):
        with open(os.path.join(input_path, filename), "r") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                intensities.append(float(row[5]))

    mean = np.mean(intensities)
    std_dev = np.std(intensities)

    print(f"Mean: {mean}, STD: {std_dev}")


def preprocess_single_frame(frame: np.array):
    resized_matrix = []
    for track_cloud in frame:
        track_cloud_len = len(track_cloud)

        # Pad or cut
        if track_cloud_len < 64:
            num_to_pad = 64 - track_cloud_len
            zero_arrays = np.zeros((num_to_pad, 5))
            padded_data = np.concatenate((track_cloud, zero_arrays), axis=0)
        else:
            padded_data = track_cloud[:64]

        # Sort
        sorted_indices = np.argsort(padded_data[:, 0])
        sorted_data = padded_data[sorted_indices]

        # Resize to matrix
        resized_matrix.append(sorted_data.reshape((8, 8, 5)))

    return np.array(resized_matrix)


# experiment_name = "final2"
# threshold = 10
# # preprocess_csv(experiment_name, threshold)
# # preprocess_npy(experiment_name, threshold)

# data = np.load("./dataset/preprocessed/final2/4.npy")


# print(data)
find_intensity_normalizers()
