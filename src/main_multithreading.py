import sys
import time
import os
import cProfile
import pstats
import multiprocessing
import threading
import numpy as np
from PyQt5.QtWidgets import QApplication
from wakepy import keep
import constants as const
from queue import Empty
from ReadDataIWR1443 import ReadIWR14xx
from Visualizer import Visualizer, ScreenAdapter
from Utils import preprocess_data, PostureEstimation
from preprocessing import format_single_frame
from Tracking import (
    TrackBuffer,
    BatchedData,
)

OFFLINE = 0
ONLINE = 1


def read_from_sensor(detObj_queue: multiprocessing.Queue):
    # Online mode
    IWR1443 = ReadIWR14xx(
        const.P_CONFIG_PATH,
        CLIport=const.P_CLI_PORT,
        Dataport=const.P_DATA_PORT,
    )
    SLEEPTIME = 0.001 * IWR1443.framePeriodicity  # Sleeping period (sec)

    while True:
        # Online mode
        dataOk, _, detObj = IWR1443.read()

        if dataOk:
            detObj_queue.put(detObj)
            # time.sleep(SLEEPTIME/2)


def track_targets(
    detObj_queue: multiprocessing.Queue,
    clusters_queue: multiprocessing.Queue,
):
    trackbuffer = TrackBuffer()
    batch = BatchedData()

    counter = 0

    while True:
        detObj = detObj_queue.get()
        now = time.time()
        trackbuffer.dt = now - trackbuffer.t
        trackbuffer.t = now

        effective_data = preprocess_data(detObj)

        if effective_data.shape[0] != 0:
            trackbuffer.track(effective_data, batch)

            # frame_matrices = np.empty(0, dtype=object)
            frame_matrices = np.array(
                [
                    format_single_frame(
                        # NOTE: The inputs are in the form of [x, y, z, x', y', z', r', s]
                        track.batch.effective_data[:, [0, 1, 2, -2, -1]]
                    )
                    for track in trackbuffer.effective_tracks
                ]
            )

            if len(frame_matrices) > 0 and counter == 0:
                clusters_queue.put(frame_matrices)

            counter = (counter + 1) % const.FB_FRAMES_SKIP


def find_keypoints(
    clusters_queue: multiprocessing.Queue,
    keypoints_queue: multiprocessing.Queue,
):
    model = PostureEstimation(const.P_MODEL_PATH)
    while True:
        clusters = clusters_queue.get()
        keypoints = model.estimate_posture(clusters)

        # Put keypoints into the queue
        keypoints_queue.put(keypoints)


def plot(clusters_queue: multiprocessing.Queue):
    model = PostureEstimation(const.P_MODEL_PATH)
    if const.SCREEN_CONNECTED:
        visual = ScreenAdapter()
    else:
        visual = Visualizer(True, True, True)

    while True:
        clusters = clusters_queue.get()
        keypoints = model.estimate_posture(clusters)
        visual.update_posture(keypoints)
        visual.draw()
        # time.sleep(0.01)


# def find_keypoints(
#     clusters_queue: multiprocessing.Queue,
#     keypoints_queue: multiprocessing.Queue,
# ):
#     model = PostureEstimation(const.P_MODEL_PATH)
#     while True:
#         clusters = clusters_queue.get()
#         keypoints = model.estimate_posture(clusters)

#         # Put keypoints into the queue
#         keypoints_queue.put(keypoints)


# def plot(keypoints_queue: multiprocessing.Queue):
#     if const.SCREEN_CONNECTED:
#         visual = ScreenAdapter()
#     else:
#         visual = Visualizer(True, True, True)

#     while True:
#         keypoints = keypoints_queue.get()  # Adjust timeout as needed
#         visual.update_posture(keypoints)
#         visual.draw()


if __name__ == "__main__":
    if const.PROFILING:
        if not os.path.exists(const.P_PROFILING_PATH):
            os.makedirs(const.P_PROFILING_PATH)

        cProfile.run("main()", f"{const.P_PROFILING_PATH}perf_stats")

        with open(f"{const.P_PROFILING_PATH}profiling_results", "w") as f:
            p = pstats.Stats(f"{const.P_PROFILING_PATH}perf_stats", stream=f)
            p.sort_stats("cumulative").print_stats()
    else:

        app = QApplication(sys.argv)

        detObj_queue = multiprocessing.Queue()
        clusters_queue = multiprocessing.Queue()
        keypoints_queue = multiprocessing.Queue()

        sensor_process = multiprocessing.Process(
            target=read_from_sensor, args=(detObj_queue,)
        )
        track_process = multiprocessing.Process(
            target=track_targets, args=(detObj_queue, clusters_queue)
        )
        # posture_process = multiprocessing.Process(
        #     target=find_keypoints, args=(clusters_queue, keypoints_queue)
        # )
        # plot_process = threading.Thread(target=plot, args=(keypoints_queue,))

        # Disable screen sleep/screensaver
        with keep.presenting():
            # Start processes
            sensor_process.start()
            track_process.start()
            # posture_process.start()
            # plot_process.start()

            plot(clusters_queue)

            # Wait for processes to finish (which they won't in this case)
            sensor_process.join()
            track_process.join()
            # posture_process.join()
            # plot_process.join()
