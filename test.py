from datetime import timedelta
import cv2
import numpy as np
import os

def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return (result + ".00").replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s


def check_if_duplicate_frames(frame1, frame2):
    if frame1 is None or frame2 is None:
        return False
    difference = cv2.subtract(frame1, frame2)
    b, g, r = cv2.split(difference)
    diff_metric = np.sum(b) + np.sum(g) + np.sum(r)
    return diff_metric < 3000

def main(video_file):
    filename, _ = os.path.splitext(video_file)
    filename += "-opencv"
    # make a folder by the name of the video file
    if not os.path.isdir(filename):
        os.mkdir(filename)
    # read the video file    
    cap = cv2.VideoCapture(video_file)
    # get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    # get the list of duration spots to save
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    # start the loop
    prev_frame = None
    print(saving_frames_durations)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for fno in range(0, total_frames, saving_frames_durations):
	    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
	    _, frame = cap.read()
        if check_if_duplicate_frames(frame, prev_frame):
            # if closest duration is less than or equals the frame duration, 
            # then save the frame
            
            frame_duration_formatted = format_timedelta(timedelta(seconds=fno))
            cv2.imwrite(os.path.join(filename, f"frame{frame_duration_formatted}.jpg"), frame) 
            # drop the duration spot from the list, since this duration spot is already saved
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # increment the frame count
        prev_frame = frame
        # print(count, closest_duration)

    

# i.e if video of duration 30 seconds, saves x frame per second = 30x frames saved in total
SAVING_FRAMES_PER_SECOND = 10

if __name__ == "__main__":
    import sys
    video_file = sys.argv[1]
    main(video_file)
