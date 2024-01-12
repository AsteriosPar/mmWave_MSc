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
TR_EXPERIMENT_FILE_READ = "person3_dop.csv"
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
V_3D_AXIS = [2.0, 6.0, 2.0]
V_SCREEN_FADE_SIZE: float = 0.2
V_BBOX_HEIGHT = 1.8

# Frames and Buffering
FB_FRAMES_SKIP = 0
FB_BUFFER_SIZE = 100

# DBScan
DB_Z_WEIGHT = 0.3
DB_RANGE_WEIGHT = 0.01
DB_EPS = 0.1
DB_MIN_SAMPLES = 18

# EKF
EKF_MAX_LIFETIME = 10

EKF_DT = 0.05
EKF_R_STD = 0.01
EKF_Q_STD = 1

# point num estimation params
EKF_A_N = 0.9
EKF_EST_POINTNUM = 100
EKF_SPREAD_LIM = [1.2, 1.2, 4, 1.4, 1.4, 1.2]  # Revise the numbers
EKF_A_SPR = 0.9  # Revise

# Gate parameter
EKF_G = 2.4


# Motion Models
class CONST_ACC_MODEL:
    EKF_DIM = [9, 6]

    # Measurement Matrix
    EKF_H = np.array(
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
    def EKF_F(mult):
        return np.array(
            [
                [1, 0, 0, (EKF_DT * mult), 0, 0, (0.5 * (EKF_DT * mult) ** 2), 0, 0],
                [0, 1, 0, 0, (EKF_DT * mult), 0, 0, (0.5 * (EKF_DT * mult) ** 2), 0],
                [0, 0, 1, 0, 0, (EKF_DT * mult), 0, 0, (0.5 * (EKF_DT * mult) ** 2)],
                [0, 0, 0, 1, 0, 0, (EKF_DT * mult), 0, 0],
                [0, 0, 0, 0, 1, 0, 0, (EKF_DT * mult), 0],
                [0, 0, 0, 0, 0, 1, 0, 0, (EKF_DT * mult)],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

    def EKF_Q_DISCR(mult):
        return block_diag(
            Q_discrete_white_noise(dim=3, dt=EKF_DT * mult, var=EKF_Q_STD),
            Q_discrete_white_noise(dim=3, dt=EKF_DT * mult, var=EKF_Q_STD),
            Q_discrete_white_noise(dim=3, dt=EKF_DT * mult, var=EKF_Q_STD),
        )


class CONST_VEL_MODEL:
    EKF_DIM = [6, 3]
    # Measurement Matrix
    EKF_H = np.eye(6)

    def STATE_VEC(init):
        return [init[0], init[1], init[2], init[3], init[4], init[5]]

    # State Transition Matrix
    def EKF_F(mult):
        return np.array(
            [
                [1, 0, 0, EKF_DT * mult, 0, 0],
                [0, 1, 0, 0, EKF_DT * mult, 0],
                [0, 0, 1, 0, 0, EKF_DT * mult],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

    def EKF_Q_DISCR(mult):
        return block_diag(
            Q_discrete_white_noise(dim=3, dt=EKF_DT * mult, var=EKF_Q_STD),
            Q_discrete_white_noise(dim=3, dt=EKF_DT * mult, var=EKF_Q_STD),
        )


MOTION_MODEL = CONST_ACC_MODEL
ENABLE_MODE = OFFLINE  # OFFLINE / ONLINE


# q2 = Q_continuous_white_noise(dim=3, dt=EKF_DT, var=EKF_Q_STD)
# EKF_Q_CONT = block_diag(q2, q2)


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
