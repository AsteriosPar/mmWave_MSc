import numpy as np
from filterpy.common import Q_discrete_white_noise, Q_continuous_white_noise
from scipy.linalg import block_diag

OFFLINE = 0
ONLINE = 1
PIXEL_TO_METERS = 0.000265

# Paths and Ports
# P_CONFIG_PATH = "./config_cases/10m_77Hz.cfg"
P_CONFIG_PATH = "./config_cases/7m.cfg"
# P_CONFIG_PATH = "./config_cases/iwr1443sdk2_4m_12hz.cfg"
P_LOG_PATH = "./dataset/log/"
P_DATA_PATH = "./dataset/"
P_PROFILING_PATH = "./profiling/"
P_CLI_PORT = "/dev/ttyACM0"
P_DATA_PORT = "/dev/ttyACM1"
P_EXPERIMENT_FILE_READ = "test2.csv"
P_EXPERIMENT_FILE_WRITE = "test2.csv"
P_CLASS = "no_luggage"


###### Scene Setup ######
# Monitor Coordinates
M_X: float = 0.08
M_Y: float = -0.42
M_Z: float = 1.5
# M_SIZE = [1920 * PIXEL_TO_METERS, 1200 * PIXEL_TO_METERS]  # Laptop
# M_SIZE = [3840 * PIXEL_TO_METERS, 2160 * PIXEL_TO_METERS]  # Monitor
M_SIZE = [1.6, 0.9]  # Monitor Approximation

M_HEIGHT = 0.8

# Sensor
S_HEIGHT = 1.8
S_TILT = -17  # degrees (-180, 180)

# Plot Parameters
V_SCALLING = 1 / 1  # Scaling parameter (only for emulating)
V_3D_AXIS = [M_SIZE[0] / V_SCALLING, 4.0, M_HEIGHT + (M_SIZE[1] / V_SCALLING)]
V_SCREEN_FADE_SIZE_MAX: float = 0.3
V_SCREEN_FADE_SIZE_MIN: float = 0.14
V_SCREEN_FADE_WEIGHT: float = (
    0.08  # square size reduction (m) per 1 meter of distance from sensor
)
V_BBOX_HEIGHT = 1.8
V_BBOX_EYESIGHT_HEIGHT = 1.75


###### Frames and Buffering #######
FB_FRAMES_SKIP = 0
FB_WRITE_BUFFER_SIZE = 100
FB_READ_BUFFER_SIZE = 100


# Number of frames per Batch
FB_FRAMES_BATCH = 4
FB_FRAMES_BATCH_STATIC = 8
FB_HEIGHT_FRAME_PERIOD = 30
FB_WIDTH_FRAME_PERIOD = 20


####### Clustering #######
# DBScan
DB_Z_WEIGHT = 0.3
DB_RANGE_WEIGHT = 0.03
DB_EPS = 0.3
DB_MIN_SAMPLES_MIN = 12

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
TR_LIFETIME_DYNAMIC = 2  # sec
TR_LIFETIME_STATIC = 3
TR_VEL_THRES = 0.1  # Velocity threshold for STATIC or DYNAMIC track
TR_GATE = 4

# Kalman
KF_R_STD = 0.1
KF_Q_STD = 1

# Initialization values
KF_P_INIT = 0.01
KF_GROUP_DISP_EST_INIT = 0.01

# Kalman estimation parameters
KF_ENABLE_EST = False
KF_A_N = 0.9
KF_EST_POINTNUM = 30
KF_SPREAD_LIM = [0.2, 0.2, 2, 1.2, 1.2, 0.2]  # Revise
KF_A_SPR = 0.9  # Revise


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


ENABLE_STATIC_CLUTTER = True
MOTION_MODEL = CONST_ACC_MODEL
PROFILING = False
SYSTEM_MODE = OFFLINE  # OFFLINE / ONLINE
SCREEN_CONNECTED = False


# q2 = Q_continuous_white_noise(dim=3, dt=FB_DT, var=KF_Q_STD)
# KF_Q_CONT = block_diag(q2, q2)
