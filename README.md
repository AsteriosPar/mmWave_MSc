# Real-time mmWave Multi-Person Pose Estimation System for Privacy-Aware Windows

This repository implements a top-down approach to Multi-Person Pose Estimation through **Multi Target Tracking (MTT)** for monitoring the presence of humans entering a scene and **Pose Estimation** on every bounding box to estimate the location of 19 human-joint keypoints. It includes functionalities to estimate targets' line-of-sight and ensures privacy protection by activating local opacities on self-fading windows. The system is specifically designed to receive radar data from the **IWR1443** millimeter-wave sensor by Texas Instruments. 

NOTE: This repository uses the [MARS model](https://github.com/SizheAn/MARS) architecture as the baseline CNN for the Posture Estimation module.

## About

This is the repository for my MSc thesis: 
[*Real-time mmWave Multi-Person Pose Estimation System for Privacy-Aware Windows*]() (not linked yet)

<p align="center">
  <img src="demo.gif" alt="animated" />
</p>

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
3. Adjust the system and scene configurations from the default directory `./src/constants.py`.

4. (Optional) For creating\logging an experiment for offline experimentation, in the directory of the local copy run the logging module.
    ```sh
    python3 ./src/DataLogging.py
    ```

5. Run the program online or offline on a logged experiment.
    ```sh
    # online
    python3 ./src/main.py

    # offline
    python3 ./src/offline_main.py
    ```

