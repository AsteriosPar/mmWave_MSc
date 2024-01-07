import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import constants as const
from ReadDataIWR1443 import ReadIWR14xx
from Visualizer import Visualizer
from Localization import apply_DBscan, apply_constraints
from Tracking import TrackBuffer, perform_tracking


def main():
    # Specify the data logging path according to the labeled class
    cur_log_path = os.path.join(const.P_LOG_PATH, const.TR_CLASS)
    if not os.path.exists(cur_log_path):
        os.makedirs(cur_log_path)

    IWR1443 = ReadIWR14xx(
        const.P_CONFIG_PATH, CLIport=const.P_CLI_PORT, Dataport=const.P_DATA_PORT
    )
    SLEEPTIME = 0.001 * IWR1443.framePeriodicity  # Sleeping period (sec)

    # Specify the parameters for the data visualization
    figure = Visualizer(const.ENABLE_2D_VIEW, const.ENABLE_3D_VIEW)

    # Control loop
    dataOk, frameNumber, detObj = IWR1443.read()
    frame_count = 0
    trackbuffer = TrackBuffer()
    # data_buffer = pd.DataFrame()

    while True:
        try:
            dataOk, frameNumber, detObj = IWR1443.read()
            if dataOk:
                # print(f"Tracks:", len(trackbuffer.tracks))
                # update the raw data scatter plot
                figure.update_raw(detObj["x"], detObj["y"], detObj["z"])

                # Apply scene constraints and static clutter removal
                effective_data = apply_constraints(detObj)

                if effective_data.shape[0] != 0:
                    if not trackbuffer.has_active_tracks():
                        clusters = apply_DBscan(effective_data)
                        trackbuffer.add_tracks(clusters)

                    else:
                        perform_tracking(effective_data, trackbuffer)

                # Update visualization graphs
                figure.update(trackbuffer)

                trackbuffer.dt_multiplier = 1

            else:
                trackbuffer.dt_multiplier += 1

                # if (
                #     frame_count % const.FB_FRAMES_SKIP == 0
                #     and const.ENABLE_DATA_LOGGING
                # ):
                #     # Prepare data for logging
                #     data = {
                #         # "Frame": frameNumber,
                #         "X": detObj["x"],
                #         "Y": detObj["y"],
                #         "Z": detObj["z"],
                #         # "Label": "A",
                #     }

                #     # Store data in the data path
                #     df = pd.DataFrame(data)
                #     df.to_csv(
                #         os.path.join(
                #             cur_log_path,
                #             f"{const.TR_EXPERIMENT_ID}_{frameNumber}.csv",
                #         ),
                #         mode="w",
                #         index=False,
                #         header=False,
                #     )
                # # data_buffer = data_buffer.append(df, ignore_index=True)
                # data_buffer = pd.concat([data_buffer, df], ignore_index=True)

                # if len(data_buffer) >= const.FB_BUFFER_SIZE:
                #     data_buffer.to_csv(
                #         cur_log_path,
                #         mode="a",
                #         index=False,
                #         header=False,
                #     )

                #     # Clear the buffer
                #     data_buffer.drop(data_buffer.index, inplace=True)

                # frame_count += 1

            time.sleep(SLEEPTIME)  # Sampling frequency of 20 Hz
        except KeyboardInterrupt:
            plt.close()
            # if const.ENABLE_DATA_LOGGING:
            #     data_buffer.to_csv(
            #         cur_log_path,
            #         mode="a",
            #         index=False,
            #         header=False,
            #     )
            del IWR1443
            break


if __name__ == "__main__":
    main()
