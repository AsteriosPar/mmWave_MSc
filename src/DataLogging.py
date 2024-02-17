import sys
import time
import os
import shutil
from threading import Thread, Event
from queue import Queue
import pandas as pd
import constants as const
from ReadDataIWR1443 import ReadIWR14xx


def query_to_overwrite(
    question="An experiment file with the same name already exists. Overwrite?\n",
):
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [y/N]: "
    default = False

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if choice == "":
            return default
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Invalid answer\n")


def read_thread(queue: Queue, IWR1443: ReadIWR14xx, SLEEPTIME, stop_event: Event):
    frame_select = const.FB_FRAMES_SKIP + 1

    try:
        while not stop_event.is_set():
            t0 = time.time()
            dataOk, frameNumber, detObj = IWR1443.read()
            if dataOk and frameNumber % frame_select == 0:
                queue.put((frameNumber, detObj))

            sys.stdout.write(f"\rFrame Number: {frameNumber}")
            sys.stdout.flush()

            t_code = time.time() - t0
            t_sleep = max(0, SLEEPTIME - t_code)
            time.sleep(t_sleep)

    except KeyboardInterrupt:
        stop_event.set()


def write_thread(queue: Queue, data_path, stop_event: Event):
    data_buffer = pd.DataFrame()
    cur_file_index = 1
    cur_file = os.path.join(data_path, f"{cur_file_index}.csv")
    frames_in_cur_file = 0
    try:
        while not stop_event.is_set():
            frameNumber, detObj = queue.get()

            # Prepare data for logging
            data = {
                "Frame": frameNumber,
                "X": detObj["x"],
                "Y": detObj["y"],
                "Z": detObj["z"],
                "Doppler": detObj["doppler"],
            }

            # # Store data in the data path
            df = pd.DataFrame(data)
            data_buffer = pd.concat([data_buffer, df], ignore_index=True)
            frames_in_cur_file += 1

            # Check if buffer size or file size limit is reached
            if (
                len(data_buffer) >= const.FB_WRITE_BUFFER_SIZE
                or frames_in_cur_file >= const.FB_EXPERIMENT_FILE_SIZE
            ):
                # Write data to CSV
                df = pd.DataFrame(data_buffer)
                df.to_csv(cur_file, mode="a", index=False, header=False)
                data_buffer.drop(data_buffer.index, inplace=True)

                # Update file index and file path if necessary
                if frames_in_cur_file >= const.FB_EXPERIMENT_FILE_SIZE:
                    frames_in_cur_file = 0
                    cur_file_index += 1
                    cur_file = os.path.join(data_path, f"{cur_file_index}.csv")

    except KeyboardInterrupt:
        stop_event.set()


def data_log_mode():
    # Create the general folder system
    if not os.path.exists(const.P_LOG_PATH):
        os.makedirs(const.P_LOG_PATH)

    # Create the logging file target
    data_path = os.path.join(
        const.P_LOG_PATH,
        const.P_EXPERIMENT_FILE_WRITE,
    )
    if os.path.exists(data_path):
        if query_to_overwrite():
            shutil.rmtree(data_path)
            os.makedirs(data_path)
        else:
            return
    else:
        os.makedirs(data_path)

    queue = Queue()
    stop_event = Event()

    IWR1443 = ReadIWR14xx(
        const.P_CONFIG_PATH, CLIport=const.P_CLI_PORT, Dataport=const.P_DATA_PORT
    )
    SLEEPTIME = 0.001 * IWR1443.framePeriodicity  # Sleeping period (sec)

    # Create separate threads for reading and writing
    read_thread_instance = Thread(
        target=read_thread, args=(queue, IWR1443, SLEEPTIME, stop_event)
    )
    write_thread_instance = Thread(
        target=write_thread, args=(queue, data_path, stop_event)
    )

    try:
        read_thread_instance.start()
        write_thread_instance.start()

        read_thread_instance.join()
        write_thread_instance.join()

    except KeyboardInterrupt:
        stop_event.set()
        read_thread_instance.join()
        write_thread_instance.join()


# Call the main function
data_log_mode()
