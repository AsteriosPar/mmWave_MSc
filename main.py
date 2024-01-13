import time
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import constants as const
from ReadDataIWR1443 import ReadIWR14xx
from Visualizer import Visualizer
from Localization import apply_DBscan, apply_constraints, batch_frames
from Tracking import TrackBuffer, perform_tracking

OFFLINE = 0
ONLINE = 1


def main():
    if const.ENABLE_MODE == OFFLINE:
        experiment_path = os.path.join(const.P_LOG_PATH, const.TR_EXPERIMENT_FILE_READ)
        if not os.path.exists(experiment_path):
            raise ValueError(f"No experiment file found in the path: {experiment_path}")

        pointclouds = {}
        with open(experiment_path, "r") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                framenum = int(row[0])
                coords = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]

                if framenum in pointclouds:
                    # Append coordinates to the existing lists
                    for key, value in zip(["x", "y", "z", "doppler"], coords):
                        pointclouds[framenum][key].append(value)
                else:
                    # If not, create a new dictionary for the framenum
                    pointclouds[framenum] = {
                        "x": [coords[0]],
                        "y": [coords[1]],
                        "z": [coords[2]],
                        "doppler": [coords[3]],
                    }

        frame_count = 0
        SLEEPTIME = 0.1  # config "frameCfg"

    else:
        IWR1443 = ReadIWR14xx(
            const.P_CONFIG_PATH, CLIport=const.P_CLI_PORT, Dataport=const.P_DATA_PORT
        )
        SLEEPTIME = 0.001 * IWR1443.framePeriodicity  # Sleeping period (sec)

    figure = Visualizer()
    trackbuffer = TrackBuffer()
    effective_data = np.empty((0, const.MOTION_MODEL.EKF_DIM[1]), dtype="float")
    batch_counter = 0

    # Control loop
    while True:
        try:
            if const.ENABLE_MODE == OFFLINE:
                # Offline mode
                frame_count += 1
                if frame_count in pointclouds:
                    dataOk = True
                    detObj = pointclouds[frame_count]
                else:
                    dataOk = False

            else:
                # Online mode
                dataOk, _, detObj = IWR1443.read()

            if dataOk:
                figure.clear()

                # update the raw data scatter plot
                figure.update_raw(detObj["x"], detObj["y"], detObj["z"])

                # Apply scene constraints and static clutter removal
                filtered_data = apply_constraints(detObj)

                effective_data, batch_counter, is_ready = batch_frames(
                    effective_data, filtered_data, batch_counter
                )

                if is_ready and effective_data.shape[0] != 0:
                    if not trackbuffer.has_active_tracks():
                        clusters = apply_DBscan(effective_data)
                        trackbuffer.add_tracks(clusters)

                    else:
                        perform_tracking(effective_data, trackbuffer)

                    trackbuffer.dt_multiplier = 1
                    effective_data = np.empty(
                        (0, const.MOTION_MODEL.EKF_DIM[1]), dtype="float"
                    )

                # Update visualization graphs
                figure.update(trackbuffer)

                figure.draw()
            else:
                trackbuffer.dt_multiplier += 1

            time.sleep(SLEEPTIME)  # Sampling frequency of 20 Hz

        except KeyboardInterrupt:
            plt.close()
            if const.ENABLE_MODE == ONLINE:
                del IWR1443
            break


if __name__ == "__main__":
    main()
