import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import constants as const
from ReadDataIWR1443 import ReadIWR14xx
from Visualizer import Visualizer
from Localization import apply_DBscan, apply_constraints


def main():
    # Specify the data logging path
    data_path = const.P_DATA_PATH
    if os.path.exists(data_path):
        os.remove(data_path)

    IWR1443 = ReadIWR14xx(
        const.P_CONFIG_PATH, CLIport=const.P_CLI_PORT, Dataport=const.P_DATA_PORT
    )
    SLEEPTIME = 0.001 * IWR1443.framePeriodicity  # Sleeping period (sec)

    # Specify the parameters for the data visualization
    figure = Visualizer(const.ENABLE_2D_VIEW, const.ENABLE_3D_VIEW)

    # Control loop
    dataOk, frameNumber, detObj = IWR1443.read()
    frame_count = 0
    data_buffer = pd.DataFrame()

    while True:
        try:
            dataOk, frameNumber, detObj = IWR1443.read()
            if dataOk:
                # update the raw data scatter plot
                figure.update_raw(detObj["x"], detObj["y"], detObj["z"])

                # Apply scene constraints and static clutter removal
                effective_data = apply_constraints(detObj)

                if len(effective_data[0]) != 0:
                    # DBScan Clustering
                    cluster_labels = apply_DBscan(effective_data[1])

                    # Update visualization graphs
                    figure.update(effective_data[1], cluster_labels)

                    if (
                        frame_count % const.FB_FRAMES_SKIP == 0
                        and const.ENABLE_DATA_LOGGING
                    ):
                        # Prepare data for logging
                        data = {
                            "Frame": frameNumber,
                            "X": detObj["x"],
                            "Y": detObj["y"],
                            "Label": "A",
                        }

                        # Store data in the data path
                        df = pd.DataFrame(data)
                        # data_buffer = data_buffer.append(df, ignore_index=True)
                        data_buffer = pd.concat([data_buffer, df], ignore_index=True)

                        if len(data_buffer) >= const.FB_BUFFER_SIZE:
                            data_buffer.to_csv(
                                data_path,
                                mode="a",
                                index=False,
                                header=False,
                            )

                            # Clear the buffer
                            data_buffer.drop(data_buffer.index, inplace=True)

                frame_count += 1

            time.sleep(SLEEPTIME)  # Sampling frequency of 20 Hz
        except KeyboardInterrupt:
            plt.close()
            if const.ENABLE_DATA_LOGGING:
                data_buffer.to_csv(
                    data_path,
                    mode="a",
                    index=False,
                    header=False,
                )
            del IWR1443
            break


if __name__ == "__main__":
    main()
