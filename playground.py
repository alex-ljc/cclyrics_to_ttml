from yt_dlp import YoutubeDL
from youtube_search import YoutubeSearch
import os
import glob
import pickle
import cv2
from ProcessCCL import *



def temp(frame1: np.ndarray, frame2: np.ndarray) -> float:
    if frame1 is None or frame2 is None:
        return 0
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    difference = cv2.subtract(frame1, frame2)
    size = frame1.shape[0] * frame1.shape[1]
    return (np.sum(difference ** 2)) / size


dir = "./frames/ENHYPEN - Bite Me"
frames = sorted([os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])
for file1, file2 in zip(frames, frames[1:]):
    frame1 = cv2.imread(file1)
    frame2 = cv2.imread(file2)
    if temp(frame1, frame2) < 1:
        print(temp(frame1, frame2), file1[len(dir) + 1:-4], file2[len(dir) + 1:-4])
        
    