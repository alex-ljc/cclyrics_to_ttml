import difflib
import itertools
import os
import pickle
import re
import sys
from datetime import timedelta
from typing import NewType

import cv2
import numpy as np
from easyocr import Reader
from lyricsgenius import Genius
from tqdm import tqdm
from webvtt import Caption, WebVTT
from youtube_search import YoutubeSearch
from yt_dlp import YoutubeDL
from typing import List
import glob

class TimedFrame:
    def __init__(self, time: timedelta, frame: np.ndarray):
        self.time = time
        self.frame = frame
    
    def __str__(self):
        return f"{self.time} {self.frame.shape}"

class TimedText:
    def __init__(self, time: timedelta, text: str):
        self.time = time
        self.text = text
        
    def __str__(self):
        return f"{self.time} {self.text}" 

def format_timedelta(td: timedelta) -> str:
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    td = td-timedelta(days=td.days)
    result = '0' + str(td)
    if "." in result:
        result, ms = result.split(".")
    else:
        return (result + ".000")
    
    ms = int(ms)
    ms = round(ms / 1e4)
    formatted = f"{result}.{ms:03}"
    return formatted

def mse_between_frames(frame1: np.ndarray, frame2: np.ndarray) -> float:
    if frame1 is None or frame2 is None:
        return 0
    
    difference = cv2.subtract(frame1, frame2)
    b, g, r = cv2.split(difference)
    size = frame1.shape[0] * frame1.shape[1]
    return (np.sum(b ** 2) + np.sum(g ** 2) + np.sum(r ** 2)) / size

def is_different_frame(frame1: np.ndarray, frame2: np.ndarray) -> bool:
    if frame1 is None or frame2 is None:
        return False
    return mse_between_frames(frame1, frame2) > 1

def ccl_to_frames(video_file: str, dir: str = '') -> List[TimedFrame]:
    if not os.path.exists(dir) and dir != '':
        os.makedirs(dir)
    
    print("Converting ccl to frames...")
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Hardcoded Crop... I should figure out a way to detect cropped boundaries
    frame_y_begin = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.8)
    frame_y_end = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.967)
    frame_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    count = 0
    unique_frames = []
    transitioning = False
    previous_frame = None
    pre_transition_frame = None


    success = 0
    while True:
        read, frame = cap.read()
        if not read:
            break
        # This methox occasionally misses when the video repositions text
        cropped = frame[frame_y_begin:frame_y_end, 0:frame_x]
        curr_time = timedelta(seconds=count / fps)
        count += 1
        
        is_prev_and_curr_different = is_different_frame(previous_frame, cropped)
        
        # Don't know why but pretransition frame always needs to be before cropped???
        is_prev_frame_too_soon = curr_time - unique_frames[-1].time > timedelta(milliseconds=750) if len(unique_frames) > 0 else True
        if is_prev_and_curr_different and not transitioning:
            transitioning = True
            pre_transition_frame = previous_frame
            # print("Found transition start with mse:", mse_between_frames(previous_frame, cropped), " at time", curr_time)
        elif not is_prev_and_curr_different and transitioning and is_prev_frame_too_soon:
            transitioning = False
            # print("Found new unique frame with mse:", mse_between_frames(previous_frame, cropped), "at time", curr_time, "and success:", success)
            cv2.imwrite(f"{dir}/{curr_time}.jpg", cropped)
            unique_frames.append(TimedFrame(timedelta(seconds=count / fps), cropped))
            success += 1
        previous_frame = cropped
        
    print("Found", len(unique_frames), "unique frames.")
    return unique_frames
        

def remove_ascii(text: str) -> str:
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

def frames_to_text(timed_frames: List[TimedFrame]) -> List[TimedText]:
    print("Reading text from frames...")
    reader = Reader(['en', 'ko'], gpu=False)
    lines = []
    for timed_frame in tqdm(timed_frames):
        image = preprocess_image(timed_frame.frame)
        # cv2.imwrite(f"./recent/{frame[0]}.jpg", image)
        line = reader.readtext(image, detail=0, paragraph=True)
        line = line[-1] if len(line) > 0 else ""
        line = remove_ascii(str(line))
        
        lines.append(TimedText(timed_frame.time, line))
    for line in lines:
        print(line)   
    return lines

def isolate_text(image: np.ndarray, leeway: int = 10) -> np.ndarray:
    if np.sum(image) < 10:
        return image[0:1, 0:1]

    first_x = None
    last_x = None
    first_y = None
    last_y = None
    for y, pixels in enumerate(image):
        for x, pixel in enumerate(pixels):
            if pixel != 0:
                first_x = min(first_x, x) if first_x is not None else x
                last_x = max(last_x, x) if last_x is not None else x
                first_y = min(first_y, y) if first_y is not None else y
                last_y = max(last_y, y) if last_y is not None else y
    
    first_x = max(0, first_x - leeway) 
    last_x = min(image.shape[1], last_x + leeway)
    first_y = max(0, first_y - leeway)
    last_y = min(image.shape[0], last_y + leeway)
    return image[first_y:last_y, first_x:last_x]

def preprocess_image(image: np.ndarray) -> np.ndarray:
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        image = isolate_text(image)
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        image = cv2.medianBlur(image,5)
    except Exception as e:
        print(str(e))
    return image

def remove_duplicate_lines(timed_lines: List[TimedText]) -> List[TimedText]:
    processed = []

    timed_lines.insert(0, TimedText(timedelta(seconds=0), ""))
    for first_timed_line, second_timed_line in zip(timed_lines, timed_lines[1:]):
        first_spaceless_line = first_timed_line.text.replace(' ', '').lower()
        second_spaceless_line = second_timed_line.text.replace(' ', '').lower()
        output_list = [li for li in difflib.ndiff(first_spaceless_line, second_spaceless_line) if li[0] != ' ']
        
        is_different = len(output_list) > len(first_spaceless_line) / 3
        if is_different:
            processed.append(second_timed_line)

    return processed

def remove_invalid_lines(timed_lines: List[TimedText]) -> List[TimedText]:
    # engDict = enchant.Dict("en_US")
    valid_lines = []
    for timed_line in timed_lines:
        valid_num_spaces = timed_line.text.count(' ') > int(len(timed_line.text) / 10)              
        # valid_num_eng_words = len([word for word in line.split() if engDict.check(word)]) > len(line.split()) / 4
        
        if valid_num_spaces:
            valid_lines.append(timed_line)
    
    return valid_lines

def spell_check(timed_lines: List[TimedText]) -> List[TimedText]:
    spell_checked_lines = []
    for timed_line in timed_lines:
        line = timed_line.text.replace(';', '')
        
        num_forward_brackets = line.count('[')
        num_backward_brackets = line.count(']')
        if num_forward_brackets != num_backward_brackets:
            line = line.replace('[', 'I')
            line = line.replace(']', 'I')
        line = line.replace('1 ', 'I ')
        line = line.replace(' 0 ', ' a ')
        line = line.replace(' [ ', ' I ')
        line = line.replace(' 1 ', ' I ')
        line = line.replace(' ] ', ' I ')
        line = line.replace(' d ', ' a ')
        line = line.replace(" ''", " '")
        line = line.replace(" 'il", " 'll")
        line = line.replace(" 'li", " 'll")
        line = line.replace(" 'ii", " 'll")
        new_line = ''
        for word in line.split():
            # Make sure word is properly capitalized
            new_line += word[0] + word[1:].lower() + ' '
        # sentences = re.findall('[A-Z][^A-Z]*', line)
        # spell_checked_line = ''.join([str(TextBlob(sentence).correct()) for sentence in sentences])
        # spell_checked_lines.append((time, spell_checked_line))
        # print("line: ", line, " vs spell checked line: ", spell_checked_line, " at time: ", time)
        spell_checked_lines.append(TimedText(timed_line.time, new_line))
    
    return spell_checked_lines

def text_to_captions(timed_lines: List[TimedText], delay: int = 700) -> WebVTT:
    vtt = WebVTT()
    timed_lines.append(TimedText(timed_lines[-1].time + timedelta(seconds=5), ""))
    for first_line, second_line in zip(timed_lines, timed_lines[1:]):
        if len(first_line.text) < 1:
            continue
        beginning = format_timedelta(first_line.time + timedelta(milliseconds=(delay)))
        end = format_timedelta(second_line.time + timedelta(milliseconds=(delay)))
        caption = Caption(beginning, end, first_line.text)
        vtt.captions.append(caption)
            
    return vtt

def add_delay_to_timed_line(timed_lines: List[TimedText], delay: int) -> List[TimedText]:
    for timed_line in timed_lines:
        timed_line.time += timedelta(milliseconds=delay)
    return timed_lines

# Shit argument design... 
def video_file_to_text(video_path: str, cache_name: str, delay: int, use_cache: bool = False, cache_path: str = '/Users/alexchen/Coding/Projects/cclyrics_to_ttml') -> WebVTT:
    # I think the way I want to design the database aspect of this is to define a cache section in a config file
    # and then have a sub folder available elsewhere
    
    # This needs to be rewritten to work with the database
    if not os.path.exists(f'{cache_path}/lines/{cache_name}.pkl') or not use_cache:
        if not os.path.exists(f'{cache_path}/frames/{cache_name}'):
            os.mkdir(f'{cache_path}/frames/{cache_name}')
            
        frames = ccl_to_frames(video_path, f'{cache_path}/frames/{cache_name}')
        lines = frames_to_text(frames)
        print(os.path.join(cache_path, 'lines', f'{cache_name}.pkl'))
        pickle.dump(lines, open(os.path.join(cache_path, 'lines', f'{cache_name}.pkl'), "wb"))
    else: 
        lines = pickle.load(open(os.path.join(cache_path, 'lines', f'{cache_name}.pkl'), "rb"))
        
    lines = spell_check(lines)
    lines = remove_duplicate_lines(lines)
    captions = text_to_captions(lines, delay)
    return captions



# Python arg parser instead of this bullshit
# Currently a fair bit of temporal information leakage
if __name__ == "__main__":
    video_file = sys.argv[1]
    song_name = sys.argv[2] if len(sys.argv) > 2 else video_file
    generate_new = len(sys.argv) > 3 and sys.argv[3] == "new"

# #1:45 Every day the same old same old Time now to toss that rule Aespa spicy seems to start a couple seconds late??? FIx bug
# # Same with 1:59
