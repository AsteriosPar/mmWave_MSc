import os
import pandas as pd
import numpy as np
import constants as const


def preprocess_csv(experiment_name, threshold):
    input_path = os.path.join(const.P_LOG_PATH, experiment_name)
    output_path = os.path.join(const.P_PREPROCESS_PATH, experiment_name)
    print("reached")

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


experiment_name = "test6"
threshold = 7
preprocess_csv(experiment_name, threshold)
