import sys
import time
import os
import cProfile
import pstats
import numpy as np
from PyQt5.QtWidgets import QApplication
from wakepy import keep
import constants as const
from ReadDataIWR1443 import ReadIWR14xx
from Visualizer import Visualizer, ScreenAdapter
from Utils import (
    preprocess_data,
    OfflineManager,
)
from preprocess import preprocess_single_frame
from Tracking import (
    TrackBuffer,
    BatchedData,
)
from PostureEstimation import PostureEstimation

OFFLINE = 0
ONLINE = 1


def main():
    if const.SYSTEM_MODE == OFFLINE:
        experiment_path = os.path.join(
            f"{const.P_LOG_PATH}{const.P_MMWAVE_DIR}", const.P_EXPERIMENT_FILE_READ
        )
        if not os.path.exists(experiment_path):
            raise ValueError(f"No experiment file found in the path: {experiment_path}")

        sensor_data = OfflineManager(experiment_path)
        SLEEPTIME = 0.1  # config "frameCfg"

    else:
        # Online mode
        IWR1443 = ReadIWR14xx(
            const.P_CONFIG_PATH, CLIport=const.P_CLI_PORT, Dataport=const.P_DATA_PORT
        )
        SLEEPTIME = 0.001 * IWR1443.framePeriodicity  # Sleeping period (sec)

    app = QApplication(sys.argv)

    if const.SCREEN_CONNECTED:
        visual = ScreenAdapter()
    else:
        visual = Visualizer(True, False, False)

    trackbuffer = TrackBuffer()
    model = PostureEstimation(const.P_MODEL_PATH)
    batch = BatchedData()

    # Disable screen sleep/screensaver
    with keep.presenting():
        # Control loop
        while True:
            try:
                t0 = time.time()
                if const.SYSTEM_MODE == OFFLINE:
                    dataOk, _, detObj = sensor_data.get_data()

                    if sensor_data.is_finished():
                        break

                    if not dataOk:
                        t_code = time.time() - t0
                        time.sleep(max(0, SLEEPTIME - t_code))

                else:
                    # Online mode
                    dataOk, _, detObj = IWR1443.read()

                if dataOk:
                    now = time.time()
                    trackbuffer.dt = now - trackbuffer.t
                    trackbuffer.t = now
                    # Apply scene constraints, translation and static clutter removal
                    effective_data = preprocess_data(detObj)

                    if effective_data.shape[0] != 0:
                        trackbuffer.track(effective_data, batch)

                        # frame_matrices = np.empty(0, dtype=object)
                        frame_matrices = np.array(
                            [
                                preprocess_single_frame(
                                    # NOTE: The inputs are in the form of [x, y, z, x', y', z', r', s]
                                    track.batch.effective_data[:, [0, 1, 2, -2, -1]]
                                )
                                for track in trackbuffer.effective_tracks
                            ]
                        )
                        frame_keypoints = model.estimate_posture(frame_matrices)

                        # visual.update_posture(frame_keypoints)

                    if const.SCREEN_CONNECTED:
                        visual.update(trackbuffer)
                    else:
                        # visual.clear()
                        # update the raw data scatter plot
                        visual.update_raw(detObj["x"], detObj["y"], detObj["z"])
                        # Update visualization graphs
                        # visual.update_bb(trackbuffer)
                        visual.draw()

                    t_code = time.time() - t0
                    t_sleep = max(0, SLEEPTIME - t_code)
                    time.sleep(t_sleep)

            except KeyboardInterrupt:
                if const.SYSTEM_MODE == ONLINE:
                    del IWR1443
                    # app.exit()
                break


if __name__ == "__main__":
    if const.PROFILING:
        if not os.path.exists(const.P_PROFILING_PATH):
            os.makedirs(const.P_PROFILING_PATH)

        cProfile.run("main()", f"{const.P_PROFILING_PATH}perf_stats")

        with open(f"{const.P_PROFILING_PATH}profiling_results", "w") as f:
            p = pstats.Stats(f"{const.P_PROFILING_PATH}perf_stats", stream=f)
            p.sort_stats("cumulative").print_stats()
    else:
        main()
