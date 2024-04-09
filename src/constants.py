import numpy as np
from filterpy.common import Q_discrete_white_noise, Q_continuous_white_noise
from scipy.linalg import block_diag

OFFLINE = 0
ONLINE = 1
PIXEL_TO_METERS = 0.000265

# Paths and Ports
P_CONFIG_PATH = "./config_cases/8.5_new.cfg"
P_MODEL_PATH = "./model/MARS.h5"
P_DATA_PATH = "./dataset"

P_LOG_PATH = f"{P_DATA_PATH}/log"
P_PREPROCESS_PATH = f"{P_DATA_PATH}/preprocessed"
P_FORMATTED_PATH = f"{P_DATA_PATH}/formatted"

P_KINECT_DIR = "/kinect/"
P_MMWAVE_DIR = "/mmWave/"

P_PROFILING_PATH = "./profiling/"

P_CLI_PORT = "/dev/ttyACM0"
P_DATA_PORT = "/dev/ttyACM1"

# Experiment specifications
P_EXPERIMENT_FILE_READ = "A41"

###### Scene Setup ######
# Sensitive Coordinates
M_X = 0.28
M_Y = -0.5
M_Z = 1.55

# Window Attributes
# M_SIZE = [1920 * PIXEL_TO_METERS, 1200 * PIXEL_TO_METERS]  # Laptop
SCREEN_SIZE = [1.6, 0.9]  # Monitor Approximation
SCREEN_HEIGHT = 2

# Sensor Attributes
S_HEIGHT = 1.6
S_TILT = -5  # degrees (-180, 180)

# Plot Parameters
V_SCALLING = 1  # Scaling parameter (only for emulating)

V_3D_AXIS = [[-2.5, 2.5], [0, 5], [0, 3]]
V_SCREEN_FADE_SIZE_MAX: float = 0.3
V_SCREEN_FADE_SIZE_MIN: float = 0.14
V_SCREEN_FADE_WEIGHT: float = (
    0.08  # square size reduction (m) per 1 meter of distance from sensor
)
V_BBOX_HEIGHT = 1.8
V_BBOX_EYESIGHT_HEIGHT = 1.75


###### Frames and Buffering #######
FB_FRAMES_SKIP = 0
FB_EXPERIMENT_FILE_SIZE = 200
FB_WRITE_BUFFER_SIZE = 40  # NOTE: must divide FB_EXPERIMENT_FILE_SIZE
FB_READ_BUFFER_SIZE = 40

# Number of frames per Batch
FB_FRAMES_BATCH = 1
FB_FRAMES_BATCH_STATIC = 1
FB_HEIGHT_FRAME_PERIOD = 30
FB_WIDTH_FRAME_PERIOD = 20


####### Clustering #######
# DBScan
DB_Z_WEIGHT = 0.4
DB_RANGE_WEIGHT = 0.03
DB_EPS = 0.3
DB_MIN_SAMPLES_MIN = 40

# Inner DBScan
DB_POINTS_THRES = 40
DB_SPREAD_THRES = 0.7
DB_INNER_EPS = 0.1
DB_INNER_MIN_SAMPLES = 8
DB_MIN_SAMPLES_MAX = 25
m = (DB_INNER_MIN_SAMPLES - DB_MIN_SAMPLES_MAX) / 10


def db_min_sample(y):
    return int(DB_MIN_SAMPLES_MAX - m * y)


###### Tracking and Kalman ######
# Tracks
TR_LIFETIME_DYNAMIC = 3  # sec
TR_LIFETIME_STATIC = 5
TR_VEL_THRES = 0.1  # Velocity threshold for STATIC or DYNAMIC track
TR_GATE = 4.5

# Kalman
KF_R_STD = 0.1
KF_Q_STD = 1

# Initialization values
KF_P_INIT = 0.1
KF_GROUP_DISP_EST_INIT = 0.1

# Kalman estimation parameters
KF_ENABLE_EST = False
KF_A_N = 0.9
KF_EST_POINTNUM = 30
KF_SPREAD_LIM = [0.2, 0.2, 2, 1.2, 1.2, 0.2]  # Revise
KF_A_SPR = 0.9  # Revise

############### Model ####################
# Intensity Normalization
INTENSITY_MU = 193
INTENSITY_STD = 252

MODEL_MIN_INPUT = 0
MODEL_DEFAULT_POSTURE = np.array(
    [
        0.00000000,
        -0.00079287,
        -0.00069028,
        -0.00386972,
        -0.18202336,
        -0.25408896,
        -0.25791826,
        0.18307304,
        0.29578214,
        0.29407474,
        -0.08050454,
        -0.11416449,
        -0.12328033,
        -0.13587935,
        0.07962164,
        0.14361644,
        0.15583284,
        0.17200874,
        -0.00076715,
        0.76999552,
        1.09064452,
        1.40205857,
        1.55136266,
        1.28932279,
        1.03600181,
        0.79945567,
        1.28655952,
        1.04830291,
        0.81176811,
        0.76708653,
        0.34282525,
        0.00000000,
        -0.07469891,
        0.77131499,
        0.37066848,
        -0.01288807,
        -0.07960824,
        1.32559470,
        0.07527616,
        0.05331851,
        0.02032234,
        0.00000000,
        0.04961569,
        0.13505063,
        0.13039448,
        0.03457985,
        0.12770589,
        0.10500648,
        0.03924988,
        0.05333718,
        0.07861458,
        -0.00562979,
        0.03460233,
        -0.00071120,
        0.06832139,
        -0.00826901,
        0.03123243,
    ]
)


# Motion Models
class CONST_ACC_MODEL:
    KF_DIM = [9, 6]

    # Measurement Matrix
    KF_H = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
        ]
    )

    def STATE_VEC(init):
        return [init[0], init[1], init[2], init[3], init[4], init[5], 0, 0, 0]

    # State Transition Matrix
    def KF_F(dt):
        return np.array(
            [
                [1, 0, 0, dt, 0, 0, (0.5 * dt**2), 0, 0],
                [0, 1, 0, 0, dt, 0, 0, (0.5 * dt**2), 0],
                [0, 0, 1, 0, 0, dt, 0, 0, (0.5 * dt**2)],
                [0, 0, 0, 1, 0, 0, dt, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, dt, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, dt],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

    def KF_Q_DISCR(dt):
        return block_diag(
            Q_discrete_white_noise(dim=3, dt=dt, var=KF_Q_STD),
            Q_discrete_white_noise(dim=3, dt=dt, var=KF_Q_STD),
            Q_discrete_white_noise(dim=3, dt=dt, var=KF_Q_STD),
        )


class CONST_VEL_MODEL:
    KF_DIM = [6, 6]
    # Measurement Matrix
    KF_H = np.eye(6)

    def STATE_VEC(init):
        return [init[0], init[1], init[2], init[3], init[4], init[5]]

    # State Transition Matrix
    def KF_F(dt):
        return np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

    def KF_Q_DISCR(dt):
        return block_diag(
            Q_discrete_white_noise(dim=3, dt=dt, var=KF_Q_STD),
            Q_discrete_white_noise(dim=3, dt=dt, var=KF_Q_STD),
        )


ENABLE_STATIC_CLUTTER = False
MOTION_MODEL = CONST_ACC_MODEL
PROFILING = False
SYSTEM_MODE = OFFLINE  # OFFLINE / ONLINE
SCREEN_CONNECTED = False


# q2 = Q_continuous_white_noise(dim=3, dt=FB_DT, var=KF_Q_STD)
# KF_Q_CONT = block_diag(q2, q2)
