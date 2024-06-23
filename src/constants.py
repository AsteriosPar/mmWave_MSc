import numpy as np
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

PIXEL_TO_METERS = 0.000265


##### General Flags #####
PROFILING = False
SCREEN_CONNECTED = True

##### Paths and Ports #####
P_CONFIG_PATH = "./config_cases/our_config_8.5m.cfg"
P_MODEL_PATH = "../trained_cases/Our_system/model/MARS.h5"
P_DATA_PATH = "./dataset"

P_LOG_PATH = f"{P_DATA_PATH}/log"
P_PREPROCESS_PATH = f"{P_DATA_PATH}/preprocessed"
P_FORMATTED_PATH = f"{P_DATA_PATH}/formatted"

P_KINECT_DIR = "/kinect/"
P_MMWAVE_DIR = "/mmWave/"

P_PROFILING_PATH = "./profiling/"

P_CLI_PORT = "/dev/ttyACM0"
P_DATA_PORT = "/dev/ttyACM1"

###### Scene Setup ######
# Sensitive Object Coordinates
M_X = 0.32
M_Y = -0.6
M_Z = 1.3

# Smart Window Attributes
# SCREEN_SIZE = [1920 * PIXEL_TO_METERS, 1200 * PIXEL_TO_METERS]  # Laptop
SCREEN_SIZE = [1.6, 1.1]  # Smart Window Size
SCREEN_HEIGHT = 1.3  # Smart Window Installation Height

# Sensor Attributes
S_HEIGHT = 1.8  # Sensor Installation Height
S_TILT = -5  # Sensor Tilt: (-180, 180)

# Plot Parameters
V_SCALLING = 1  # Scaling parameter (only for emulating)

V_3D_AXIS = [[-2.5, 2.5], [0, 5], [0, 3]]
V_SCREEN_FADE_SIZE_MAX: float = 0.3
V_SCREEN_FADE_SIZE_MIN: float = 0.2
V_SCREEN_FADE_WEIGHT: float = (
    0.08  # square size reduction m) per 1 meter of distance from sensor
)
V_BBOX_HEIGHT = 1.8
V_BBOX_EYESIGHT_HEIGHT = 1.75


###### Experiment Logging #######
FB_FRAMES_SKIP = 0
FB_EXPERIMENT_FILE_SIZE = 200
FB_WRITE_BUFFER_SIZE = 40  # NOTE: must divide FB_EXPERIMENT_FILE_SIZE
FB_READ_BUFFER_SIZE = 40


####### Clustering #######
# Ringbuffer
FB_FRAMES_BATCH = 2
FB_FRAMES_BATCH_STATIC = 2

# DBScan
DB_Z_WEIGHT = 0.4
DB_RANGE_WEIGHT = 0.03
DB_EPS = 0.3
DB_MIN_SAMPLES_MIN = 35

# Inner DBScan (Currently inactive)
DB_POINTS_THRES = 40
DB_SPREAD_THRES = 0.7
DB_INNER_EPS = 0.1
DB_INNER_MIN_SAMPLES = 8
DB_MIN_SAMPLES_MAX = 25


###### Tracking and Kalman ######
# Tracks
TR_MAX_TRACKS = 4
TR_LIFETIME_DYNAMIC = 3  # sec
TR_LIFETIME_STATIC = 7  # sec
TR_VEL_THRES = 0.12  # Velocity threshold for STATIC or DYNAMIC track
TR_GATE = 4.5

# Kalman Noise Variances
KF_R_STD = 0.1
KF_Q_STD = 1

# Initialization values
KF_P_INIT = 0.1
KF_GROUP_DISP_EST_INIT = 0.1

# GTRACK pointnum & spread estimation
KF_ENABLE_EST = False
KF_A_N = 0.9
KF_EST_POINTNUM = 10
KF_SPREAD_LIM = [0.2, 0.2, 2, 1.2, 1.2, 0.2]
KF_A_SPR = 0.9

############### Model ####################
# Intensity Normalization
INTENSITY_MU = 27.0187
INTENSITY_STD = 70.351

MODEL_MIN_INPUT = 0
MODEL_DEFAULT_POSTURE = np.array(
    [
        0.0000,
        -0.0007,
        -0.0006,
        -0.0038,
        -0.1820,
        -0.2540,
        -0.2579,
        0.1830,
        0.2957,
        0.2940,
        -0.0805,
        -0.1141,
        -0.1232,
        -0.1358,
        0.0796,
        0.1436,
        0.1558,
        0.1720,
        -0.0007,
        0.7699,
        1.0906,
        1.4020,
        1.5513,
        1.2893,
        1.0360,
        0.7994,
        1.2865,
        1.0483,
        0.8117,
        0.7670,
        0.3428,
        0.0000,
        -0.0746,
        0.7713,
        0.3706,
        -0.0128,
        -0.0796,
        1.3255,
        0.0752,
        0.0533,
        0.0203,
        0.0000,
        0.0496,
        0.1350,
        0.1303,
        0.0345,
        0.1277,
        0.1050,
        0.0392,
        0.0533,
        0.0786,
        -0.0056,
        0.0346,
        -0.0007,
        0.0683,
        -0.0082,
        0.0312,
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


MOTION_MODEL = CONST_ACC_MODEL
