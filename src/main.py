import sys
import time
import os
import cProfile
import pstats
from PyQt5.QtWidgets import QApplication
from wakepy import keep
import constants as const
from ReadDataIWR1443 import ReadIWR14xx
from Visualizer import Visualizer, ScreenAdapter
from Utils import (
    preprocess_data,
    read_next_frames,
)
from Tracking import (
    TrackBuffer,
    BatchedData,
)
from PostureEstimation import PostureEstimation

OFFLINE = 0
ONLINE = 1


def main():
    if const.SYSTEM_MODE == OFFLINE:
        experiment_path = os.path.join(const.P_LOG_PATH, const.P_EXPERIMENT_FILE_READ)
        if not os.path.exists(experiment_path):
            raise ValueError(f"No experiment file found in the path: {experiment_path}")

        pointclouds, pointer, last_frame = read_next_frames(experiment_path)

        frame_count = 0
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
        visual = Visualizer(True, True, True)

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
                    # If last_frame is None the offline experiment has reached its end or the experiment file was not found
                    if last_frame is None:
                        break

                    frame_count += 1
                    # If the read buffer is parsed, read more frames from the experiment file
                    if frame_count > last_frame:
                        pointclouds, pointer, last_frame = read_next_frames(
                            experiment_path, pointer
                        )
                    # Get the data associated to the current frame
                    if frame_count in pointclouds:
                        dataOk = True
                        detObj = pointclouds[frame_count]
                    else:
                        t_code = time.time() - t0
                        time.sleep(max(0, SLEEPTIME - t_code))
                        dataOk = False

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

                        # Estimate posture
                        # keypoints = []
                        for track in trackbuffer.effective_tracks:
                            keypoints = model.estimate_posture(
                                track.batch.effective_data
                            )
                            visual.update_posture(keypoints)
                            break

                    if const.SCREEN_CONNECTED:
                        visual.update(trackbuffer)
                    else:
                        visual.clear()
                        # update the raw data scatter plot
                        visual.update_raw(detObj["x"], detObj["y"], detObj["z"])
                        # Update visualization graphs
                        visual.update_bb(trackbuffer)
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
