import numpy as np
import pandas as pd
import shutil
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
from wakepy import keep

KINECT_Z = 0.8
KINECT_X = 0.22
RELATIVE_ENABLED = True


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
    # print(len(pairs))
    return pairs


def relative_coordinates(absolute_coords: np.array, reference: np.array):
    # NOTE: Keep the z-axis intact to not add noise
    return np.array(
        [
            point - [reference[0], reference[1], 0, 0, 0, 0, 0, 0]
            for point in absolute_coords
        ]
    )


def filter_kinect_frames(pairs, invalid_frames, experiment, centroids):
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

        valid_counter = 0
        for row in reader:
            if any(int(row[1]) == f_pair[1] for f_pair in pairs) and not any(
                int(row[1]) == inv_frame1 for inv_frame1 in invalid_kinect_frames
            ):

                translated_row = translate_kinect(row)

                if RELATIVE_ENABLED:
                    # current_centroid = centroids[valid_counter]
                    # translated_row = relative_kinect(translated_row, current_centroid)
                    translated_row = static_kinect(translated_row)

                writer.writerow(translated_row)
                valid_counter += 1


def translate_kinect(row):
    ang_rad = np.radians(6.5)
    z, y = 0, 0
    for i in range(2, len(row) - 1):
        # For all x coords
        if i % 3 == 2:
            row[i] = str(float(row[i]) + KINECT_X)
            z = float(row[i + 1])
            y = float(row[i + 2])

        # For all z coords:
        elif i % 3 == 0:
            row[i] = str(
                y * np.sin(ang_rad) + float(row[i]) * np.cos(ang_rad) + KINECT_Z
            )

        # For all y coords:
        else:
            row[i] = str(float(row[i]) * np.cos(ang_rad) - z * np.sin(ang_rad))

    return row


def relative_kinect(row, centroid):
    for i in range(2, len(row) - 1):
        # For all x coords
        if i % 3 == 2:
            row[i] = str(float(row[i]) - centroid[0])

        # For all y coords:
        elif i % 3 == 1:
            row[i] = str(float(row[i]) - centroid[1])

    return row


def static_kinect(row):
    # NOTE: the static skeleton has its lower back on the x=0 plane and its left foot on the z=0 plane
    # Lower back x: row[2], Left foot z: row[39]
    x_abs = float(row[2])
    y_abs = float(row[13])
    z_abs = float(row[39])
    for i in range(2, len(row) - 1):
        # For all x coords
        if i % 3 == 2:
            row[i] = str(float(row[i]) - x_abs)

        # For all y coords
        elif i % 3 == 1:
            row[i] = str(float(row[i]) - y_abs)

        # For all z coords:
        else:
            row[i] = str(float(row[i]) - z_abs)

    return row


def preprocess_dataset():

    experiments_directory = f"{const.P_LOG_PATH}{const.P_MMWAVE_DIR}"
    for experiment in os.listdir(experiments_directory):
        print("new exp")

        frame_pairs = pair(experiment)

        input_dir = os.path.join(f"{const.P_LOG_PATH}{const.P_MMWAVE_DIR}", experiment)
        output_dir = f"{const.P_PREPROCESS_PATH}{const.P_MMWAVE_DIR}/{experiment}"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
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
        centroids = []
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
                            track_points = trackbuffer.effective_tracks[
                                0
                            ].batch.effective_data
                            if len(track_points) > 0:

                                valid_frame = True

                                if RELATIVE_ENABLED:
                                    track_points = relative_coordinates(
                                        track_points,
                                        trackbuffer.effective_tracks[
                                            0
                                        ].cluster.centroid,
                                    )
                                    centroids.append(
                                        trackbuffer.effective_tracks[
                                            0
                                        ].cluster.centroid[:2],
                                    )

                                # Save effective data in a .csv
                                data = {
                                    "Frame": framenum,
                                    "X": track_points[:, 0],
                                    "Y": track_points[:, 1],
                                    "Z": track_points[:, 2],
                                    "Doppler": track_points[:, 6],
                                    "Intensity": track_points[:, 7],
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
                                    or frames_in_cur_file
                                    >= const.FB_EXPERIMENT_FILE_SIZE
                                ):
                                    # Write data to CSV
                                    df = pd.DataFrame(data_buffer)
                                    df.to_csv(
                                        cur_file, mode="a", index=False, header=False
                                    )
                                    data_buffer.drop(data_buffer.index, inplace=True)

                                    # Update file index and file path if necessary
                                    if (
                                        frames_in_cur_file
                                        >= const.FB_EXPERIMENT_FILE_SIZE
                                    ):
                                        frames_in_cur_file = 0
                                        cur_file_index += 1
                                        cur_file = os.path.join(
                                            output_dir, f"{cur_file_index}.csv"
                                        )

            if not valid_frame:
                invalid_frames.append(framenum)

        # Write remaining data to CSV
        df = pd.DataFrame(data_buffer)
        df.to_csv(cur_file, mode="a", index=False, header=False)

        filter_kinect_frames(frame_pairs, invalid_frames, experiment, centroids)


def find_intensity_normalizers(sets):
    intensities = []
    for mode in sets:
        experiments_directory = f"{const.P_PREPROCESS_PATH}{const.P_MMWAVE_DIR}{mode}/"
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

    ##################

    ##################

    # Resize to matrix
    return sorted_data.reshape((8, 8, 5))


def format_mmwave_to_npy(mean, std_dev, mode):
    experiments_directory = f"{const.P_PREPROCESS_PATH}{const.P_MMWAVE_DIR}{mode}/"
    output_path = f"{const.P_FORMATTED_PATH}{const.P_MMWAVE_DIR}"
    main_list = []

    experiments = os.listdir(experiments_directory)
    experiments_sorted = sorted(experiments, key=extract_parts)

    for experiment in experiments_sorted:
        experiment_path = os.path.join(experiments_directory, experiment)
        filenames = os.listdir(experiment_path)
        filenames_sorted = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))
        for filename in filenames_sorted:
            print(f"doing {experiment}, {filename}...")
            with open(os.path.join(experiment_path, filename), "r") as file:
                df = pd.read_csv(file, header=None)
                current_frame = None
                frame_array = []
                for _, row in df.iterrows():
                    if current_frame is None:
                        current_frame = row[0]
                        frame_array.append(row[1:6])

                    elif current_frame == row[0]:
                        frame_array.append(row[1:6])

                    else:
                        main_list.append(
                            format_single_frame(np.array(frame_array), mean, std_dev)
                        )

                        frame_array = []
                        current_frame = row[0]
                        frame_array.append(row[1:6])

                main_list.append(
                    format_single_frame(np.array(frame_array), mean, std_dev)
                )
    print(np.array(main_list).shape)
    # Save to output .npy file
    np.save(
        os.path.join(output_path, f"{mode}_mmWave.npy"),
        np.array(main_list),
    )


def format_kinect_to_npy(mode):
    experiments_directory = f"{const.P_PREPROCESS_PATH}{const.P_KINECT_DIR}{mode}/"
    experiments = os.listdir(experiments_directory)
    experiments_sorted = sorted(experiments, key=extract_parts)
    output_path = f"{const.P_FORMATTED_PATH}{const.P_KINECT_DIR}"

    main_list = []
    for experiment in experiments_sorted:
        with open(os.path.join(experiments_directory, experiment), "r") as exp:

            frames = pd.read_csv(exp, header=None)
            for _, frame in frames.iterrows():
                main_list.append(np.array(frame[2:59]).reshape(-1, 3).T.flatten())

    print(np.array(main_list).shape)
    # Save to output .npy file
    np.save(
        os.path.join(output_path, f"{mode}_labels.npy"),
        np.array(main_list),
    )


def format_dataset():
    sets = ["training", "validate", "testing"]
    mean, std_dev = find_intensity_normalizers(sets)

    with keep.presenting():

        for set_mode in sets:
            # Preprocess .csvs into numpy arrays and save them in one file
            format_mmwave_to_npy(mean, std_dev, set_mode)
            format_kinect_to_npy(set_mode)


def extract_parts(filename):
    base, ext = os.path.splitext(filename)
    numeric_part = "".join(filter(str.isdigit, base))
    alpha_part = "".join(filter(str.isalpha, base))
    return int(numeric_part), alpha_part, ext


def split_sets():
    validate_prefix = ["A5"]
    testing_prefix = ["A4"]

    kinect_directory = f"{const.P_PREPROCESS_PATH}{const.P_KINECT_DIR}/"
    mmwave_directory = f"{const.P_PREPROCESS_PATH}{const.P_MMWAVE_DIR}/"

    directories = [kinect_directory, mmwave_directory]

    for directory in directories:
        experiments = os.listdir(directory)

        shutil.rmtree(f"{directory}/training")
        shutil.rmtree(f"{directory}/validate")
        shutil.rmtree(f"{directory}/testing")
        os.makedirs(f"{directory}/training")
        os.makedirs(f"{directory}/validate")
        os.makedirs(f"{directory}/testing")

        for experiment in experiments:
            if (
                experiment.find("training") == -1
                and experiment.find("validate") == -1
                and experiment.find("testing") == -1
            ):
                if any(experiment.find(prefix) != -1 for prefix in validate_prefix):
                    shutil.move(
                        f"{directory}/{experiment}",
                        f"{directory}/validate/{experiment}",
                    )

                elif any(experiment.find(prefix) != -1 for prefix in testing_prefix):
                    shutil.move(
                        f"{directory}/{experiment}",
                        f"{directory}/testing/{experiment}",
                    )
                else:
                    shutil.move(
                        f"{directory}/{experiment}",
                        f"{directory}/training/{experiment}",
                    )


# preprocess_dataset()
# split_sets()
# format_dataset()
