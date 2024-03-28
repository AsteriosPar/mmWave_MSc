import numpy as np
import pandas as pd
import os
import os
import csv
import constants as const
from Utils import (
    preprocess_data,
    OfflineManager,
)
from Tracking import (
    TrackBuffer,
    BatchedData,
)


def pair(experiment):
    kinect_input = os.path.join(
        f"{const.P_LOG_PATH}{const.P_KINECT_DIR}", f"{experiment}.csv"
    )
    mmwave_input = os.path.join(f"{const.P_LOG_PATH}{const.P_MMWAVE_DIR}", experiment)

    df2 = pd.read_csv(kinect_input, header=None)
    pairs = []
    filenames = os.listdir(mmwave_input)
    filenames_sorted = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))
    for filename in filenames_sorted:
        with open(os.path.join(mmwave_input, filename), "r") as file:
            df1 = pd.read_csv(file, header=None)
            unique_frames = df1.drop_duplicates(subset=0)

            for _, row1 in unique_frames.iterrows():
                timestamp1 = row1[6]
                closest_row = df2.iloc[(df2[0] - timestamp1).abs().argsort()[:1]]
                timestamp2 = closest_row.iloc[0, 0]

                if abs(timestamp1 - timestamp2) < 20:
                    pairs.append((int(row1[0]), closest_row.iloc[0, 1]))
    print(pairs)
    return pairs


def filter_kinect_frames(pairs, invalid_frames, experiment):
    input_file = os.path.join(
        f"{const.P_LOG_PATH}{const.P_KINECT_DIR}", f"{experiment}.csv"
    )
    output_file = os.path.join(
        f"{const.P_PREPROCESS_PATH}{const.P_KINECT_DIR}", f"{experiment}.csv"
    )

    invalid_kinect_frames = []
    for inv_frame in invalid_frames:
        for pair in pairs:
            if pair[0] == inv_frame:
                invalid_kinect_frames.append(pair[1])

    with open(input_file, "r", newline="") as infile, open(
        output_file, "w", newline=""
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            if any(int(row[1]) == f_pair[1] for f_pair in pairs) and not any(
                int(row[1]) == inv_frame1 for inv_frame1 in invalid_kinect_frames
            ):
                writer.writerow(row)


def preprocess_dataset(experiment=const.P_EXPERIMENT_FILE_READ):
    frame_pairs = pair(experiment)

    input_dir = os.path.join(f"{const.P_LOG_PATH}{const.P_MMWAVE_DIR}", experiment)
    output_dir = os.path.join(
        f"{const.P_PREPROCESS_PATH}{const.P_MMWAVE_DIR}", experiment
    )
    os.makedirs(output_dir)
    data_buffer = pd.DataFrame()
    frames_in_cur_file = 0
    cur_file_index = 1
    cur_file = os.path.join(output_dir, f"{cur_file_index}.csv")

    trackbuffer = TrackBuffer()
    batch = BatchedData()
    sensor_data = OfflineManager(input_dir)

    first_iter = True
    invalid_frames = []
    while not sensor_data.is_finished():

        valid_frame = False
        dataOk, framenum, detObj = sensor_data.get_data()

        if any(framenum == pair[0] for pair in frame_pairs):
            if dataOk:
                if first_iter:
                    trackbuffer.dt = 0.1
                    first_iter = False
                else:
                    trackbuffer.dt = detObj["posix"][0] / 1000 - trackbuffer.t

                trackbuffer.t = detObj["posix"][0] / 1000
                effective_data = preprocess_data(detObj)

                if effective_data.shape[0] != 0:
                    trackbuffer.track(effective_data, batch)

                    if len(trackbuffer.effective_tracks) > 0:
                        single_target_points = trackbuffer.effective_tracks[
                            0
                        ].batch.effective_data
                        if single_target_points.shape[0] > 0:

                            valid_frame = True

                            # Save effective data in a .csv
                            data = {
                                "Frame": framenum,
                                "X": single_target_points[:, 0],
                                "Y": single_target_points[:, 1],
                                "Z": single_target_points[:, 2],
                                "Doppler": single_target_points[:, 6],
                                "Intensity": single_target_points[:, 7],
                            }

                            # Store data in the data path
                            df = pd.DataFrame(data)
                            data_buffer = pd.concat(
                                [data_buffer, df], ignore_index=True
                            )
                            frames_in_cur_file += 1

                            # Check if buffer size or file size limit is reached
                            if (
                                len(data_buffer) >= const.FB_WRITE_BUFFER_SIZE
                                or frames_in_cur_file >= const.FB_EXPERIMENT_FILE_SIZE
                            ):
                                # Write data to CSV
                                df = pd.DataFrame(data_buffer)
                                df.to_csv(cur_file, mode="a", index=False, header=False)
                                data_buffer.drop(data_buffer.index, inplace=True)

                                # Update file index and file path if necessary
                                if frames_in_cur_file >= const.FB_EXPERIMENT_FILE_SIZE:
                                    frames_in_cur_file = 0
                                    cur_file_index += 1
                                    cur_file = os.path.join(
                                        output_dir, f"{cur_file_index}.csv"
                                    )

        if not valid_frame:
            invalid_frames.append(framenum)

    filter_kinect_frames(frame_pairs, invalid_frames, experiment)


def find_intensity_normalizers():
    intensities = []
    experiments_directory = f"{const.P_PREPROCESS_PATH}{const.P_MMWAVE_DIR}"
    for experiment in os.listdir(experiments_directory):
        input_path = os.path.join(experiments_directory, experiment)
        for filename in os.listdir(input_path):
            with open(os.path.join(input_path, filename), "r") as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    intensities.append(float(row[5]))

    mean = np.mean(intensities)
    std_dev = np.std(intensities)

    print(f"Mean: {mean}, STD: {std_dev}")
    return mean, std_dev


def format_single_frame(
    track_cloud: np.array, mean=const.INTENSITY_MU, std_dev=const.INTENSITY_STD
):
    track_cloud_len = len(track_cloud)

    # Normalize Intensity
    track_cloud[:, 4] = (track_cloud[:, 4] - mean) / std_dev

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
    return sorted_data.reshape((8, 8, 5))


def format_mmwave_to_npy(mean, std_dev):
    experiments_directory = f"{const.P_PREPROCESS_PATH}{const.P_MMWAVE_DIR}"
    output_path = f"{const.P_FORMATTED_PATH}{const.P_MMWAVE_DIR}"
    main_array = np.empty((0, 8, 8, 5))

    for experiment in os.listdir(experiments_directory):
        experiment_path = os.path.join(experiments_directory, experiment)
        for filename in os.listdir(experiment_path):
            with open(os.path.join(experiment_path, filename), "r") as file:
                df = pd.read_csv(file, header=None)
                unique_frames = df.drop_duplicates(subset=0)

                for _, row_unique in unique_frames.iterrows():
                    frame_array = []
                    frame_found = False
                    for _, row in df.iterrows():
                        if row_unique[0] == row[0]:
                            frame_found = True
                            frame_array.append(row[1:6])

                        elif frame_found:
                            break

                    main_array = np.concatenate(
                        (
                            main_array,
                            [format_single_frame(np.array(frame_array), mean, std_dev)],
                        ),
                        axis=0,
                    )
    print(main_array.shape)
    # Save to output .npy file
    np.save(
        os.path.join(output_path, "training_mmWave.npy"),
        main_array,
    )


def format_kinect_to_npy():
    experiments_directory = f"{const.P_PREPROCESS_PATH}{const.P_KINECT_DIR}"
    output_path = f"{const.P_FORMATTED_PATH}{const.P_KINECT_DIR}"

    main_list = []
    for experiment in os.listdir(experiments_directory):
        with open(os.path.join(experiments_directory, experiment), "r") as exp:

            frames = pd.read_csv(exp, header=None)
            for _, frame in frames.iterrows():
                main_list.append(np.array(frame[2:59]).reshape(-1, 3).T.flatten())

    print(np.array(main_list).shape)
    # Save to output .npy file
    np.save(
        os.path.join(output_path, "training_labels.npy"),
        np.array(main_list),
    )


def format_dataset():
    mean, std_dev = find_intensity_normalizers()
    # Preprocess .csvs into numpy arrays and save them in one file
    format_mmwave_to_npy(mean, std_dev)
    format_kinect_to_npy()


# preprocess_dataset()
format_dataset()
