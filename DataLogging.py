import sys
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import constants as const
from ReadDataIWR1443 import ReadIWR14xx
from Visualizer import Visualizer


def query_yes_no(
    question="An experiment file with the same name already exists. Overwrite?\n",
):
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [y/N]: "
    default = False

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if choice == "":
            return default
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Invalid answer\n")


def data_log_mode():
    # Create the general folder system
    if not os.path.exists(const.P_LOG_PATH):
        os.makedirs(const.P_LOG_PATH)

    # Create the logging file target
    data_path = os.path.join(
        const.P_LOG_PATH,
        const.P_EXPERIMENT_FILE_WRITE,
    )
    if os.path.exists(data_path):
        if query_yes_no():
            os.remove(data_path)
        else:
            return

    data_buffer = pd.DataFrame()
    frame_count = 0

    IWR1443 = ReadIWR14xx(
        const.P_CONFIG_PATH, CLIport=const.P_CLI_PORT, Dataport=const.P_DATA_PORT
    )
    SLEEPTIME = 0.001 * IWR1443.framePeriodicity  # Sleeping period (sec)

    # figure = Visualizer()

    # Control loop
    while True:
        try:
            t0 = time.time()
            dataOk, frameNumber, detObj = IWR1443.read()
            if dataOk:
                # figure.update_raw(detObj["x"], detObj["y"], detObj["z"])
                # figure.draw()

                if frame_count % (const.FB_FRAMES_SKIP + 1) == 0:
                    # Prepare data for logging
                    data = {
                        "Frame": frameNumber,
                        "X": detObj["x"],
                        "Y": detObj["y"],
                        "Z": detObj["z"],
                        "Doppler": detObj["doppler"],
                    }

                    # Store data in the data path
                    df = pd.DataFrame(data)
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

            # time.sleep(SLEEPTIME)  # Sampling frequency of 20 Hz
            t_code = time.time() - t0
            t_sleep = max(0, SLEEPTIME - t_code)
            time.sleep(t_sleep)

        except KeyboardInterrupt:
            # plt.close()
            data_buffer.to_csv(
                data_path,
                mode="a",
                index=False,
                header=False,
            )
            del IWR1443
            break


data_log_mode()
