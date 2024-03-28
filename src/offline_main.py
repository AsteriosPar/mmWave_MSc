import time
import os
import constants as const
from Utils import (
    preprocess_data,
    OfflineManager,
)
from Tracking import (
    TrackBuffer,
    BatchedData,
)


def main():
    experiment_path = os.path.join(const.P_LOG_PATH, const.P_EXPERIMENT_FILE_READ)
    if not os.path.exists(experiment_path):
        raise ValueError(f"No experiment file found in the path: {experiment_path}")

    SLEEPTIME = 0.1
    trackbuffer = TrackBuffer()
    batch = BatchedData()
    sensor_data = OfflineManager(experiment_path)

    while True:
        try:
            t0 = time.time()
            dataOk, _, detObj = sensor_data.get_data()
            if dataOk:
                now = time.time()
                trackbuffer.dt = now - trackbuffer.t
                trackbuffer.t = now
                effective_data = preprocess_data(detObj)

                if effective_data.shape[0] != 0:
                    trackbuffer.track(effective_data, batch)

                    # frame_matrices = np.array(
                    #     [
                    #         preprocess_single_frame(
                    #             # NOTE: The inputs are in the form of [x, y, z, x', y', z', r', s]
                    #             track.batch.effective_data[:, [0, 1, 2, -2, -1]]
                    #         )
                    #         for track in trackbuffer.effective_tracks
                    #     ]
                    # )
                    # frame_keypoints = model.estimate_posture(frame_matrices)

            # Check if experiment is finished
            if sensor_data.is_finished():
                break

            # Emulate the sensor frequency
            t_code = time.time() - t0
            time.sleep(max(0, SLEEPTIME - t_code))

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
