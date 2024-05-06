import numpy as np
import pandas as pd
import shutil
import os
import csv
import constants as const
from Utils import (
    normalize_data,
    OfflineManager,
    format_single_frame_pre,
    format_single_frame,
    relative_coordinates,
    format_single_frame_lite,
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
    # Lower back x: row[2], feet z: row[39], row[51]
    x_abs = float(row[2])
    y_abs = float(row[13])
    z_abs = min(float(row[39]), float(row[51]))
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
                    effective_data = normalize_data(detObj)

                    if effective_data.shape[0] != 0:
                        trackbuffer.track(effective_data, batch)

                        if len(trackbuffer.effective_tracks) > 0:
                            track_points = trackbuffer.effective_tracks[
                                0
                            ].batch.effective_data

                            if (
                                trackbuffer.effective_tracks[0].lifetime == 0
                                and len(track_points) > 0
                            ):
                                valid_frame = True

                                if RELATIVE_ENABLED:
                                    track_points = relative_coordinates(
                                        list(
                                            trackbuffer.effective_tracks[0].batch.buffer
                                        ),
                                        trackbuffer.effective_tracks[
                                            0
                                        ].cluster.centroid,
                                    )

                                final_frames = format_single_frame_lite(track_points)

                                # Save effective data in a .csv
                                data = {
                                    "Frame": framenum,
                                    "X": final_frames[:, 0],
                                    "Y": final_frames[:, 1],
                                    "Z": final_frames[:, 2],
                                    "Doppler": final_frames[:, 3],
                                    "Intensity": final_frames[:, 4],
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

                else:
                    batch.pop_frame()

            if not valid_frame:
                invalid_frames.append(framenum)

        # Write remaining data to CSV
        df = pd.DataFrame(data_buffer)
        df.to_csv(cur_file, mode="a", index=False, header=False)

        filter_kinect_frames(frame_pairs, invalid_frames, experiment)


def find_intensity_normalizers(sets):
    intensities = []
    for mode in sets:
        experiments_directory = f"{const.P_PREPROCESS_PATH}{const.P_MMWAVE_DIR}{mode}/"
        for experiment in os.listdir(experiments_directory):
            if experiment.find("N_") == -1:
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


def format_mmwave_to_npy(mean, std_dev, mode):
    experiments_directory = f"{const.P_PREPROCESS_PATH}{const.P_MMWAVE_DIR}{mode}/"
    output_path = f"{const.P_FORMATTED_PATH}{const.P_MMWAVE_DIR}"
    main_list = []

    experiments = os.listdir(experiments_directory)
    experiments_sorted = sorted(experiments, key=extract_parts)

    for experiment in experiments_sorted:
        print(f"doing {experiment}")
        experiment_path = os.path.join(experiments_directory, experiment)
        filenames = os.listdir(experiment_path)
        filenames_sorted = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))
        for filename in filenames_sorted:
            with open(os.path.join(experiment_path, filename), "r") as file:
                # csv.reader(file)
                reader = csv.reader(file)
                rows = list(reader)

                # df = pd.read_csv(file, header=None)
                current_frame = None
                frame_array = []
                # for _, row in df.iterrows():
                for row in rows:
                    if current_frame is None:
                        current_frame = int(row[0])
                        frame_array.append([float(i) for i in row[1:6]])

                    elif current_frame == int(row[0]):
                        frame_array.append([float(i) for i in row[1:6]])

                    else:
                        main_list.append(
                            format_single_frame_pre(
                                np.array(frame_array, dtype=np.float32), mean, std_dev
                            )
                        )

                        frame_array = []
                        current_frame = int(row[0])
                        frame_array.append([float(i) for i in row[1:6]])

                main_list.append(
                    format_single_frame_pre(
                        np.array(frame_array, dtype=np.float32), mean, std_dev
                    )
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
    validate_prefix = ["A5", "B3", "B9"]
    testing_prefix = ["A4", "B4", "B10"]

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


def add_noise():
    mean = 0.0
    std = 0.044
    sets = ["training", "validate", "testing"]

    # mmWave
    for mode in sets:
        experiments_directory = f"{const.P_PREPROCESS_PATH}{const.P_MMWAVE_DIR}{mode}/"
        for experiment in os.listdir(experiments_directory):
            input_path = os.path.join(experiments_directory, experiment)
            distorted_experiment = f"N_{experiment}"
            distorted_path = os.path.join(experiments_directory, distorted_experiment)
            if os.path.exists(distorted_path):
                shutil.rmtree(distorted_path)
            os.mkdir(distorted_path)
            for filename in os.listdir(input_path):
                with open(os.path.join(input_path, filename), "r") as file:
                    reader = csv.reader(file)
                    rows = list(reader)

                # Modify the data
                for row in rows:
                    for i in range(1, 4):  # Assuming columns 2, 3, 4
                        if float(row[i]) != 0:
                            row[i] = str(
                                float(row[i]) + np.random.normal(loc=mean, scale=std)
                            )

                # Write the modified data to a new CSV file
                with open(
                    os.path.join(distorted_path, filename), "w", newline=""
                ) as file:
                    writer = csv.writer(file)
                    writer.writerows(rows)

    # Kinect
    for mode in sets:
        experiments_directory = f"{const.P_PREPROCESS_PATH}{const.P_KINECT_DIR}{mode}/"
        for experiment in os.listdir(experiments_directory):
            new_file_name = f"N_{experiment}"
            new_file_path = os.path.join(experiments_directory, new_file_name)
            original_file_path = os.path.join(experiments_directory, experiment)
            shutil.copyfile(original_file_path, new_file_path)


preprocess_dataset()
split_sets()
add_noise()
format_dataset()
