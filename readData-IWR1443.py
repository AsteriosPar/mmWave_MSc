import time
import numpy as np
import serial
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

""" This module is taken from https://github.com/FmmW-Group/IWR1443-Python-API/blob/main/README.md"""


class ReadIWR14xx(object):
    def __init__(self, configFileName, CLIport="/dev/ttyACM0", Dataport="/dev/ttyACM1"):
        self.configFileName = configFileName
        # Windows COM10 COM9; Raspberry Pi /dev/ttyACM0 /dev/ttyACM1
        self.CLIport = serial.Serial(CLIport, 115200)
        self.Dataport = serial.Serial(Dataport, 921600)
        self.byteBuffer = np.zeros(2**15, dtype="uint8")
        self.byteBufferLength = 0
        self.configParameters = self.__parseConfigFile()
        self.__serialConfig()
        # Constants
        # self.OBJ_STRUCT_SIZE_BYTES = 12
        # self.BYTE_VEC_ACC_MAX_SIZE = 2**15
        self.MMWDEMO_UART_MSG_DETECTED_POINTS = 1
        # self.MMWDEMO_UART_MSG_RANGE_PROFILE   = 2
        self.maxBufferSize = 2**15
        self.magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

    def read(self):
        # Initialize variables
        magicOK = False  # Checks if magic number has been read
        dataOK = 0  # Checks if the data has been read correctly
        frameNumber = 0
        detObj = {}
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2**8, 2**16, 2**24]

        readBuffer = self.Dataport.read(self.Dataport.in_waiting)
        byteVec = np.frombuffer(readBuffer, dtype="uint8")
        byteCount = len(byteVec)
        # print(byteCount)

        # Check that the buffer is not full, and then add the data to the buffer
        if (self.byteBufferLength + byteCount) < self.maxBufferSize:
            self.byteBuffer[
                self.byteBufferLength : self.byteBufferLength + byteCount
            ] = byteVec
            self.byteBufferLength = self.byteBufferLength + byteCount

        # Check that the buffer has some data
        if self.byteBufferLength > 16:
            # Check for all possible locations of the magic word
            possibleLocs = np.where(
                self.byteBuffer[0 : (self.byteBufferLength - 8)] == self.magicWord[0]
            )[0]

            # Confirm that is the beginning of the magic word and store the index in startIdx
            startIdx = None
            for loc in possibleLocs[::-1]:
                check = self.byteBuffer[loc : loc + 8]
                if np.all(check == self.magicWord):
                    startIdx = loc
                    break

            # Check that startIdx is not empty
            if startIdx is not None:
                # Remove the data before the first start index
                self.byteBuffer[: self.byteBufferLength - startIdx] = self.byteBuffer[
                    startIdx : self.byteBufferLength
                ]
                self.byteBufferLength = self.byteBufferLength - startIdx

                # Check that there have no errors with the byte buffer length
                # if self.byteBufferLength < 0:
                #     self.byteBufferLength = 0

                if self.byteBufferLength > 16:
                    # Read the total packet length
                    totalPacketLen = np.matmul(self.byteBuffer[12 : 12 + 4], word)
                    # Check that all the packet has been read
                    if self.byteBufferLength >= totalPacketLen:
                        magicOK = True

        # If magicOK is equal to 1 then process the message
        if magicOK:
            # Read the header
            magicNumber = self.byteBuffer[0:8]
            version = format(np.matmul(self.byteBuffer[8:12], word), "x")
            totalPacketLen = np.matmul(self.byteBuffer[12:16], word)
            platform = format(np.matmul(self.byteBuffer[16:20], word), "x")
            frameNumber = np.matmul(self.byteBuffer[20:24], word)
            timeCpuCycles = np.matmul(self.byteBuffer[24:28], word)
            numDetectedObj = np.matmul(self.byteBuffer[28:32], word)
            numTLVs = np.matmul(self.byteBuffer[32:36], word)
            # print(numDetectedObj, numTLVs, self.byteBufferLength)

            # UNCOMMENT IN CASE OF SDK 2
            # subFrameNumber = np.matmul(byteBuffer[idX:idX+4],word)
            # idX += 4
            idX = 36
            # Read the TLV messages
            # for tlvIdx in range(numTLVs):
            if numDetectedObj > 0:
                # Check the header of the TLV message
                tlv_type = np.matmul(self.byteBuffer[idX : idX + 4], word)
                idX += 4
                tlv_length = np.matmul(self.byteBuffer[idX : idX + 4], word)
                idX += 4

                # Read the data depending on the TLV message
                if tlv_type == self.MMWDEMO_UART_MSG_DETECTED_POINTS:
                    # word array to convert 4 bytes to a 16 bit number
                    word = [1, 2**8]
                    tlv_numObj = np.matmul(self.byteBuffer[idX : idX + 2], word)
                    idX += 2
                    tlv_xyzQFormat = 2 ** np.matmul(
                        self.byteBuffer[idX : idX + 2], word
                    )
                    idX += 2

                    # Initialize the arrays
                    rangeIdx = np.zeros(tlv_numObj, dtype="int16")
                    dopplerIdx = np.zeros(tlv_numObj, dtype="int16")
                    peakVal = np.zeros(tlv_numObj, dtype="int16")
                    x = np.zeros(tlv_numObj, dtype="int16")
                    y = np.zeros(tlv_numObj, dtype="int16")
                    z = np.zeros(tlv_numObj, dtype="int16")

                    for objectNum in range(tlv_numObj):
                        # Read the data for each object
                        rangeIdx[objectNum] = np.matmul(
                            self.byteBuffer[idX : idX + 2], word
                        )
                        idX += 2
                        dopplerIdx[objectNum] = np.matmul(
                            self.byteBuffer[idX : idX + 2], word
                        )
                        idX += 2
                        peakVal[objectNum] = np.matmul(
                            self.byteBuffer[idX : idX + 2], word
                        )
                        idX += 2
                        x[objectNum] = np.matmul(self.byteBuffer[idX : idX + 2], word)
                        idX += 2
                        y[objectNum] = np.matmul(self.byteBuffer[idX : idX + 2], word)
                        idX += 2
                        z[objectNum] = np.matmul(self.byteBuffer[idX : idX + 2], word)
                        idX += 2

                    # Make the necessary corrections and calculate the rest of the data
                    rangeVal = rangeIdx * self.configParameters["rangeIdxToMeters"]
                    dopplerIdx[
                        dopplerIdx > (self.configParameters["numDopplerBins"] / 2 - 1)
                    ] = (
                        dopplerIdx[
                            dopplerIdx
                            > (self.configParameters["numDopplerBins"] / 2 - 1)
                        ]
                        - 65535
                    )
                    dopplerVal = (
                        dopplerIdx * self.configParameters["dopplerResolutionMps"]
                    )
                    # x[x > 32767] = x[x > 32767] - 65536
                    # y[y > 32767] = y[y > 32767] - 65536
                    # z[z > 32767] = z[z > 32767] - 65536
                    x = x / tlv_xyzQFormat
                    y = y / tlv_xyzQFormat
                    z = z / tlv_xyzQFormat

                    # Store the data in the detObj dictionary
                    detObj = {
                        "numObj": tlv_numObj,
                        "rangeIdx": rangeIdx,
                        "range": rangeVal,
                        "dopplerIdx": dopplerIdx,
                        "doppler": dopplerVal,
                        "peakVal": peakVal,
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                    dataOK = 1

            # Remove already processed data
            if idX > 0 and self.byteBufferLength > idX:
                # shiftSize = totalPacketLen
                self.byteBuffer[
                    : self.byteBufferLength - totalPacketLen
                ] = self.byteBuffer[totalPacketLen : self.byteBufferLength]
                # self.byteBuffer[self.byteBufferLength - shiftSize:] = np.zeros(len(self.byteBuffer[self.byteBufferLength - shiftSize:]),dtype = 'uint8')
                self.byteBufferLength = self.byteBufferLength - totalPacketLen

                # Check that there are no errors with the buffer length
                # if self.byteBufferLength < 0:
                #     self.byteBufferLength = 0

        return dataOK, frameNumber, detObj

    def __parseConfigFile(self):
        configParameters = (
            {}
        )  # Initialize an empty dictionary to store the configuration parameters
        # Read the configuration file and send it to the board
        config = [line.rstrip("\r\n") for line in open(self.configFileName)]
        for i in config:
            # Split the line
            splitWords = i.split(" ")

            # Hard code the number of antennas, change if other configuration is used
            numRxAnt = 4
            numTxAnt = 3

            # Get the information about the profile configuration
            if "profileCfg" in splitWords[0]:
                startFreq = int(float(splitWords[2]))
                idleTime = int(splitWords[3])
                rampEndTime = float(splitWords[5])
                freqSlopeConst = float(splitWords[8])
                numAdcSamples = int(splitWords[10])
                numAdcSamplesRoundTo2 = 1

                while numAdcSamples > numAdcSamplesRoundTo2:
                    numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2

                digOutSampleRate = int(splitWords[11])

            # Get the information about the frame configuration
            elif "frameCfg" in splitWords[0]:
                chirpStartIdx = int(splitWords[1])
                chirpEndIdx = int(splitWords[2])
                numLoops = int(splitWords[3])
                numFrames = int(splitWords[4])
                self.framePeriodicity = float(splitWords[5])

        # Combine the read data to obtain the configuration parameters
        numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
        configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
        configParameters["numRangeBins"] = numAdcSamplesRoundTo2
        configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (
            2 * freqSlopeConst * 1e12 * numAdcSamples
        )
        configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
            2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"]
        )
        configParameters["dopplerResolutionMps"] = 3e8 / (
            2
            * startFreq
            * 1e9
            * (idleTime + rampEndTime)
            * 1e-6
            * configParameters["numDopplerBins"]
            * numTxAnt
        )
        configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (
            2 * freqSlopeConst * 1e3
        )
        configParameters["maxVelocity"] = 3e8 / (
            4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt
        )

        return configParameters

    def __serialConfig(self):
        config = [line.rstrip("\r\n") for line in open(self.configFileName)]
        for i in config:
            self.CLIport.write((i + "\n").encode())
            print(i)
            time.sleep(0.01)

    def __del__(self):
        self.CLIport.write(("sensorStop\n").encode())
        self.CLIport.close()
        self.Dataport.close()


def update(x, y, z, scatter, ax2):
    # Update the data in the 3D scatter plot

    scatter._offsets3d = (x, y, z)

    center = (-np.mean(x), np.mean(z))

    ax2.clear()
    # Plot the filled square with updated center coordinates and alpha
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    square = Rectangle(
        (center[0] - 0.25, center[1] - 0.25), 0.5, 0.5, alpha=0.5, color="b"
    )
    ax2.add_patch(square)

    plt.draw()
    plt.pause(0.1)  # Pause for a short time to allow for updating


def main():
    configFileName = "/home/sergio/Desktop/thesis/IWR1443-Python-API/radarconfig/iwr1443sdk2_4m_12hz.cfg"
    IWR1443 = ReadIWR14xx(
        configFileName, CLIport="/dev/ttyACM0", Dataport="/dev/ttyACM1"
    )
    sleeptime = 0.001 * IWR1443.framePeriodicity

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(121, projection="3d")
    scatter = ax.scatter([], [], [])

    # Create a 2D plot for the square
    ax2 = fig.add_subplot(122)

    plt.show(block=False)  # Set block=False to allow continuing execution

    dataOk, frameNumber, detObj = IWR1443.read()
    while True:
        try:
            dataOk, frameNumber, detObj = IWR1443.read()
            if dataOk:
                update(detObj["x"], detObj["y"], detObj["z"], scatter, ax2)

                data = np.array(
                    [
                        detObj["x"],
                        detObj["y"],
                        detObj["z"],
                        detObj["doppler"],
                        detObj["peakVal"],
                    ]
                ).transpose(1, 0)
                print(frameNumber, np.shape(data))
            else:
                print(0)

            time.sleep(sleeptime)  # Sampling frequency of 20 Hz
        except KeyboardInterrupt:
            plt.close()
            del IWR1443
            break


if __name__ == "__main__":
    main()
