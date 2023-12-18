# Paths and Ports
P_CONFIG_PATH = "./config_cases/iwr1443sdk2_4m_12hz.cfg"
P_DATA_PATH = "./data/training_data.csv"
P_CLI_PORT = "/dev/ttyACM0"
P_DATA_PORT = "/dev/ttyACM1"

# Frames and Buffering
FB_FRAMES_SKIP = 5
FB_BUFFER_SIZE = 100

# Scene constraints and Clutter Removal
C_RANGE_MIN = 0.1
C_RANGE_MAX = 12
C_DOPPLER_THRES = 0

# DBScan
DB_Z_WEIGHT = 0.3
DB_EPS = 0.5
DB_MIN_SAMPLES = 5

# Enable actions
ENABLE_2D_VIEW = True
ENABLE_3D_VIEW = True
ENABLE_DATA_LOGGING = False
