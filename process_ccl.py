import sys
from datetime import timedelta
import cv2
import numpy as np
from easyocr import Reader
from tqdm import tqdm
import os
import itertools
from spellchecker import SpellChecker
from textblob import TextBlob
import pytesseract
import difflib
import pickle
from webvtt import WebVTT, Caption
import enchant

def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    td = td-timedelta(days=td.days)
    result = '0' + str(td)
    
    try:
        result, ms = result.split(".")
    except ValueError:
        return (result + ".000")
    ms = int(ms)
    ms = round(ms / 1e4)
    
    formatted = f"{result}.{ms:03}"
    return formatted

def mse_between_frames(frame1, frame2):
    if frame1 is None or frame2 is None:
        return 0
    difference = cv2.subtract(frame1, frame2)
    b, g, r = cv2.split(difference)
    size = frame1.shape[0] * frame1.shape[1]
    return (np.sum(b ** 2) + np.sum(g ** 2) + np.sum(r ** 2)) / size

# def mse_between_frames(frame1, frame2):
#     if frame1 is None or frame2 is None:
#         return 0
#     frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#     frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#     diff = cv2.subtract(frame1, frame2)
#     size = frame1.shape[0] * frame1.shape[1]
#     return np.sum(diff ** 2) / size

def ccl_to_frames(video_file, dir = './recent'):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    prev_frame = None
    count = 0
    
    frame_x_begin = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.56)
    frame_x_end = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_y = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   
    frames = [] 
    # If i see a change skip the next two frames and insert the future frame
    while True:
        is_read, frame = cap.read()
        if not is_read:
            break
        
        frame_duration = count / fps
        # Crop needs to not be hard coded
        frame = frame[frame_x_begin:frame_x_end, 0:frame_y]

        if mse_between_frames(prev_frame, frame) > 0.5 or prev_frame is None: 
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            prev_frame = frame
            print(f"Writing frame {frame_duration_formatted}")
            cv2.imwrite(os.path.join(dir, f"{frame_duration_formatted}.jpg"), frame)
            frames.append((timedelta(seconds=frame_duration), frame))
            
        count += 1
    return frames

def remove_ascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

def frames_to_text(frames):
    reader = Reader(['en', 'ko'], gpu=False)
    lines = []
    for frame in tqdm(frames):
        image = preprocess_image(frame[1])
        line = reader.readtext(image, detail=0, paragraph=True)
        line = line[-1] if len(line) > 0 else ""
        line = remove_ascii(line)

        lines.append((frame[0], line))
    
    return lines

def formatted_time_to_time_delta(formatted_time):
    time = formatted_time.split(":")
    hours = int(time[0])
    minutes = int(time[1])
    seconds = int(time[2]).split(".")[0]
    milliseconds = int(time[2]).split(".")[1]
    
    return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    image = cv2.medianBlur(image,5)
    return image

# Kinda bad code design to only be able to pass timed_lines and not lines???
def remove_duplicate_lines(timed_lines):
    processed = []

    # Check whetehr the [1:] in first bugs it
    for second_timed_line, first_timed_line in zip(timed_lines[1:], timed_lines):
        first_spaceless_line = first_timed_line[1].replace(' ', '')
        second_spaceless_line = second_timed_line[1].replace(' ', '')
        output_list = [li for li in difflib.ndiff(first_spaceless_line, second_spaceless_line) if li[0] != ' ']
        
        valid_diff = len(output_list) > len(first_spaceless_line) / 3
        if valid_diff:
            processed.append(first_timed_line)

    return processed

def remove_invalid_lines(timed_lines):
    engDict = enchant.Dict("en_US")
    
    valid_lines = []
    for _, line in timed_lines:
        valid_num_spaces = line.count(' ') > int(len(line) / 10)              
        valid_num_eng_words = len([word for word in line.split() if engDict.check(word)]) > len(line.split()) / 4
        # instead of rejection I should be adding on thats why its ending early I think
    
    return valid_lines

def text_to_captions(lines, delay):
    vtt = WebVTT()
    for line1, line2 in zip(lines, lines[1:]):
        # Add 0.7 seconds cos ccl is badly timed
        beginning = format_timedelta(line1[0] + timedelta(milliseconds=delay))
        end = format_timedelta(line2[0] + timedelta(milliseconds=delay))
        caption = Caption(beginning, end, line1[1])
        
        vtt.captions.append(caption)
    return vtt

if __name__ == "__main__":
    video_file = sys.argv[1]
    frames = ccl_to_frames(video_file)
    lines = frames_to_text(frames)
    pickle.dump(lines, open("spicy.pkl", "wb"))
    
    lines = pickle.load(open("spicy.pkl", "rb"))
    lines = remove_duplicate_lines(lines)
    lines = remove_invalid_lines(lines)
    captions = text_to_captions(lines, 700)
    captions.save("captions.vtt")
