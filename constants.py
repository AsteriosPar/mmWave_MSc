import numpy as np
import math
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

# Training
TR_EXPERIMENT_FILE_READ = "person4_dop.csv"
TR_EXPERIMENT_FILE_WRITE = "person1.csv"
TR_CLASS = "no_luggage"

# Monitor Coordinates
M_X = 0.6
M_Y = -1
M_Z = 0.6

# Sensor
S_HEIGHT = 0.8
S_TILT = 10  # degrees (-180, 180)

# Visualization Parameters
V_3D_AXIS = [4, 5.0, 3]
V_SCREEN_FADE_SIZE: float = 0.2
V_BBOX_HEIGHT = 1.8

# Frames and Buffering
FB_FRAMES_SKIP = 0
FB_BUFFER_SIZE = 100

# DBScan
DB_Z_WEIGHT = 0.3
DB_RANGE_WEIGHT = 0.03
DB_EPS = 0.25
DB_MIN_SAMPLES = 25

# Kalman Filter
KF_MAX_LIFETIME = 10
KF_DT = 0.05
KF_R_STD = 0.01
KF_Q_STD = 1

# point num estimation params
KF_A_N = 0.9
KF_EST_POINTNUM = 100
KF_SPREAD_LIM = [1.2, 1.2, 4, 1.4, 1.4, 1.2]  # Revise the numbers
KF_A_SPR = 0.9  # Revise

# Gate parameter
KF_G = 2.2


# Motion Models
class CONST_ACC_MODEL:
    DIM = [9, 6]

    # Measurement Matrix
    H = np.array(
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
    def F(mult):
        return np.array(
            [
                [1, 0, 0, (KF_DT * mult), 0, 0, (0.5 * (KF_DT * mult) ** 2), 0, 0],
                [0, 1, 0, 0, (KF_DT * mult), 0, 0, (0.5 * (KF_DT * mult) ** 2), 0],
                [0, 0, 1, 0, 0, (KF_DT * mult), 0, 0, (0.5 * (KF_DT * mult) ** 2)],
                [0, 0, 0, 1, 0, 0, (KF_DT * mult), 0, 0],
                [0, 0, 0, 0, 1, 0, 0, (KF_DT * mult), 0],
                [0, 0, 0, 0, 0, 1, 0, 0, (KF_DT * mult)],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

    def Q_DISCR(mult):
        return block_diag(
            Q_discrete_white_noise(dim=3, dt=KF_DT * mult, var=KF_Q_STD),
            Q_discrete_white_noise(dim=3, dt=KF_DT * mult, var=KF_Q_STD),
            Q_discrete_white_noise(dim=3, dt=KF_DT * mult, var=KF_Q_STD),
        )


class CONST_VEL_MODEL:
    DIM = [6, 3]
    # Measurement Matrix
    H = np.eye(6)

    def STATE_VEC(init):
        return [init[0], init[1], init[2], init[3], init[4], init[5]]

    # State Transition Matrix
    def F(mult):
        return np.array(
            [
                [1, 0, 0, KF_DT * mult, 0, 0],
                [0, 1, 0, 0, KF_DT * mult, 0],
                [0, 0, 1, 0, 0, KF_DT * mult],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

    def Q_DISCR(mult):
        return block_diag(
            Q_discrete_white_noise(dim=3, dt=KF_DT * mult, var=KF_Q_STD),
            Q_discrete_white_noise(dim=3, dt=KF_DT * mult, var=KF_Q_STD),
        )


# Modes
ENABLE_MODE = OFFLINE  # OFFLINE / ONLINE
MOTION_MODEL = CONST_ACC_MODEL

# q2 = Q_continuous_white_noise(dim=3, dt=KF_DT, var=KF_Q_STD)
# KF_Q_CONT = block_diag(q2, q2)


# def jacobian_matrix(state_vec):
#     r = math.sqrt(state_vec[0] ** 2 + state_vec[1] ** 2 + state_vec[2] ** 2)
#     return np.array(
#         [
#             [1, 0, 0, 0, 0, 0],
#             [0, 1, 0, 0, 0, 0],
#             [0, 0, 1, 0, 0, 0],
#             [
#                 (
#                     state_vec[1]
#                     * (state_vec[3] * state_vec[1] - state_vec[4] * state_vec[0])
#                     + state_vec[2](
#                         state_vec[3] * state_vec[2] - state_vec[5] * state_vec[0]
#                     )
#                 )
#                 / (r**3),
#                 (
#                     state_vec[0]
#                     * (state_vec[4] * state_vec[0] - state_vec[3] * state_vec[1])
#                     + state_vec[2](
#                         state_vec[4] * state_vec[2] - state_vec[5] * state_vec[1]
#                     )
#                 )
#                 / (r**3),
#                 (
#                     state_vec[0]
#                     * (state_vec[5] * state_vec[0] - state_vec[3] * state_vec[2])
#                     + state_vec[1](
#                         state_vec[5] * state_vec[1] - state_vec[4] * state_vec[2]
#                     )
#                 )
#                 / (r**3),
#                 state_vec[0] / r,
#                 state_vec[1] / r,
#                 state_vec[2] / r,
#             ],
#         ]
#     )
