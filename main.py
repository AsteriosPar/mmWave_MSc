import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ReadDataIWR1443 import ReadIWR14xx
from Visualizer import Visualizer


def main():
    configFileName = "./config_cases/iwr1443sdk2_4m_12hz.cfg"
    IWR1443 = ReadIWR14xx(
        configFileName, CLIport="/dev/ttyACM0", Dataport="/dev/ttyACM1"
    )
    sleeptime = 0.001 * IWR1443.framePeriodicity

    file_path = "./data/training_data.csv"
    if os.path.exists(file_path):
        os.remove(file_path)

    figure = Visualizer(enable_2d=True)

    dataOk, frameNumber, detObj = IWR1443.read()
    while True:
        try:
            dataOk, frameNumber, detObj = IWR1443.read()
            if dataOk:
                # update(detObj["x"], detObj["y"], detObj["z"], scatter, ax2)
                figure.update(detObj["x"], detObj["y"], detObj["z"])

                # Sample DataFrame
                data = {
                    "Frame": frameNumber,
                    "X": detObj["x"],
                    "Y": detObj["y"],
                    "Label": "A",
                }

                df = pd.DataFrame(data)
                df.to_csv(
                    file_path,
                    mode="a",
                    index=False,
                    header=not os.path.exists(file_path),
                )

            time.sleep(sleeptime)  # Sampling frequency of 20 Hz
        except KeyboardInterrupt:
            plt.close()
            del IWR1443
            break


if __name__ == "__main__":
    main()
