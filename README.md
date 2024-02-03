# Real-time mmWave Eye Localization and Tracking System for Privacy-Aware Windows

This repository implements **Multi Target Tracking (MTT)** for monitoring the presence of humans entering a scene. It includes functionalities to estimate eye localization and ensures privacy protection by activating local opacities on self-fading windows. The system is specifically designed to receive radar data from the **IWR1443** millimeter-wave sensor.

## About

This is the repository for my MSc thesis: 
[*Real-time mmWave Eye Localization and Tracking System for Privacy-Aware Windows*]()


## Getting Started

### Installation and Execution


1. Clone this repository.
   ```sh
   git clone https://github.com/AsteriosPar/mmWave_MSc
   ```

2. Install Dependencies.
   ```sh
   pip install -r requirements.txt
   ```
3. Adjust the system and scene configurations from the default directory `./src/constants.py`. More information on the system configurations on the next section. 

4. (Optional) For creating\logging an experiment for offline experimentation, in the directory of the local copy run the logging module.
    ```sh
    python3 ./src/DataLogging.py
    ```

5. Run the program.
    ```sh
    python3 ./src/main.py
    ```

## Configurations

This is a brief walk-through of the most basic configuration parameters.

### Scene Parameters
Set up the scene by specifiying the coordinates of the sensitive object, the window size and installation height, as well as the sensor mount height and tilt. 

**NOTE:** The coordinate system is established with the assumption that the point (0,0,0) is situated at ground level, precisely beneath the center of the installed window, which exists on the y=0 plane.

```python
###### Scene Setup ######
# Sensitive Coordinates
M_X = 0.08
M_Y = -0.42
M_Z = 1.5

# Window Attributes
M_SIZE = [1.6, 0.9]
M_HEIGHT = 0.8

# Sensor Attributes
S_HEIGHT = 1.6
S_TILT = -17  # degrees (-180, 180)
```

### Offline experimentation
in the Offline mode, experiment files are needed. Set the names of the experiment files to log or read from in the below fields.  
```python
P_EXPERIMENT_FILE_READ = "test1.csv"
P_EXPERIMENT_FILE_WRITE = "test2.csv"
```

### System Parameters
Through the System Parameters you can toggle between Online and Offline mode as well as choose whether or not a self-fading window is connected. In the later case, a 3D representation of the pointcloud and tracks is plotted.
```python
SYSTEM_MODE = OFFLINE  # OFFLINE / ONLINE
SCREEN_CONNECTED = False
```
