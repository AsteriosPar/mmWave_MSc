import numpy as np
from filterpy.common import Q_discrete_white_noise, Q_continuous_white_noise
from scipy.linalg import block_diag

OFFLINE = 0
ONLINE = 1

# Paths and Ports
P_CONFIG_PATH = "./config_cases/iwr1443sdk2_4m_12hz.cfg"
P_LOG_PATH = "./dataset/log/"
P_DATA_PATH = "./dataset/"
P_CLI_PORT = "/dev/ttyACM0"
P_DATA_PORT = "/dev/ttyACM1"
P_EXPERIMENT_FILE_READ = "circular_movement.csv"
P_EXPERIMENT_FILE_WRITE = "circular_resolution_check.csv"
P_CLASS = "no_luggage"


###### Scene Setup ######
# Monitor Coordinates
M_X: float = -0.6
M_Y: float = -1
M_Z: float = 0.6

# Sensor
S_HEIGHT = 1.7
S_TILT = -10  # degrees (-180, 180)

# Plot Parameters
V_3D_AXIS = [3.0, 6.0, 2.0]
V_SCREEN_FADE_SIZE_MAX: float = 0.5
V_SCREEN_FADE_SIZE_MIN: float = 0.2
V_SCREEN_FADE_WEIGHT: float = (
    0.08  # square size reduction (m) per 1 meter of distance from sensor
)
V_BBOX_HEIGHT = 1.8
V_BBOX_EYESIGHT_HEIGHT = 1.75


###### Frames and Buffering #######
FB_FRAMES_SKIP = 0
FB_BUFFER_SIZE = 100
FB_DT = 0.1


####### Clustering #######
# DBScan
DB_Z_WEIGHT = 0.3
DB_RANGE_WEIGHT = 0.01
DB_EPS = 0.2
DB_MIN_SAMPLES = 20

DB_INNER_EPS = 0.5
DB_INNER_MIN_SAMPLES = 10


# Number of frames per Batch
FB_FRAMES_BATCH = 6


###### Tracking and Kalman ######
# Tracks
TR_LIFETIME_DYNAMIC = 2.5  # sec
TR_LIFETIME_STATIC = 10
TR_VEL_THRES = 0.1  # Velocity threshold for STATIC or DYNAMIC track
TR_GATE = 4.5

# Kalman
KF_R_STD = 10
KF_Q_STD = 1

# Initialization values
KF_P_INIT = 0.001
KF_GROUP_DISP_EST_INIT = 0.001

# Kalman estimation parameters
KF_ENABLE_EST = False
KF_A_N = 0.9
KF_EST_POINTNUM = 30
KF_SPREAD_LIM = [0.7, 0.7, 2, 1.4, 1.4, 1.2]  # Revise
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


MOTION_MODEL = CONST_ACC_MODEL
ENABLE_MODE = OFFLINE  # OFFLINE / ONLINE
ENABLE_STATIC_CLUTTER = False


# q2 = Q_continuous_white_noise(dim=3, dt=FB_DT, var=KF_Q_STD)
# KF_Q_CONT = block_diag(q2, q2)
