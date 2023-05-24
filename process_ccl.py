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
import re

def format_timedelta(td: timedelta) -> str:
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

def mse_between_frames(frame1: np.ndarray, frame2: np.ndarray) -> float:
    if frame1 is None or frame2 is None:
        return 0
    difference = cv2.subtract(frame1, frame2)
    b, g, r = cv2.split(difference)
    size = frame1.shape[0] * frame1.shape[1]
    return (np.sum(b ** 2) + np.sum(g ** 2) + np.sum(r ** 2)) / size

def is_different_frame(frame1: np.ndarray, frame2: np.ndarray) -> bool:
    return mse_between_frames(frame1, frame2) > 0.5

def ccl_to_frames(video_file: str, dir = './recent'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    print("Converting ccl to frames...")
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_x_begin = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.54)
    frame_x_end = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.965)
    frame_y = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
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
        
        cropped = frame[frame_x_begin:frame_x_end, 0:frame_y]
        count += 1
        previous_frame_and_cropped_different = is_different_frame(previous_frame, cropped)
        # Don't know why but pretransition frame always needs to be before cropped???
        cropped_and_pre_transition_frame_different = is_different_frame(pre_transition_frame, cropped)
        
        if previous_frame_and_cropped_different and not transitioning:
            transitioning = True
            pre_transition_frame = previous_frame
            print("Found transition start with mse:", mse_between_frames(previous_frame, cropped), " at time", timedelta(seconds=count / fps))
        elif not previous_frame_and_cropped_different and cropped_and_pre_transition_frame_different and transitioning:
            transitioning = False
            print("Found new unique frame with mse:", mse_between_frames(previous_frame, cropped), "at time", timedelta(seconds=count / fps), "and success:", success)
            unique_frames.append((timedelta(seconds=count / fps), cropped))
            cv2.imwrite(os.path.join(dir, f"{format_timedelta(timedelta(seconds=count))}.jpg"), cropped)    
            success += 1
        
        previous_frame = cropped
    return unique_frames
        

def remove_ascii(text: str) -> str:
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

def frames_to_text(frames):
    print("Reading text from frames...")
    reader = Reader(['en', 'ko'], gpu=False)
    lines = []
    for frame in tqdm(frames):
        image = preprocess_image(frame[1])
        line = reader.readtext(image, detail=0, paragraph=True)
        line = line[-1] if len(line) > 0 else ""
        line = remove_ascii(line)
        lines.append((frame[0], line))
    
    return lines

def formatted_time_to_time_delta(formatted_time: str) -> timedelta:
    time = formatted_time.split(":")
    hours = int(time[0])
    minutes = int(time[1])
    seconds = int(time[2]).split(".")[0]
    milliseconds = int(time[2]).split(".")[1]
    
    return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    image = cv2.medianBlur(image,5)
    return image

# Kinda bad code design to only be able to pass timed_lines and not lines???
def remove_duplicate_lines(timed_lines):
    processed = []

    # Check whetehr the [1:] in first bugs it
    timed_lines.insert(0, (timedelta(seconds=0), ""))
    for first_timed_line, second_timed_line in zip(timed_lines, timed_lines[1:]):
        first_spaceless_line = first_timed_line[1].replace(' ', '')
        second_spaceless_line = second_timed_line[1].replace(' ', '')
        output_list = [li for li in difflib.ndiff(first_spaceless_line, second_spaceless_line) if li[0] != ' ']
        
        is_different = len(output_list) > len(first_spaceless_line) / 3
        if is_different:
            processed.append(second_timed_line)

    return processed

def remove_invalid_lines(timed_lines):
    engDict = enchant.Dict("en_US")
    
    valid_lines = []
    for time, line in timed_lines:
        valid_num_spaces = line.count(' ') > int(len(line) / 10)              
        valid_num_eng_words = len([word for word in line.split() if engDict.check(word)]) > len(line.split()) / 4
        
        if valid_num_spaces and valid_num_eng_words:
            valid_lines.append((time, line))
    
    return valid_lines

def spell_check(timed_lines):
    spell_checked_lines = []
    for time, line in timed_lines:
        line = line.replace(';', '')
        # sentences = re.findall('[A-Z][^A-Z]*', line)
        # spell_checked_line = ''.join([str(TextBlob(sentence).correct()) for sentence in sentences])
        # spell_checked_lines.append((time, spell_checked_line))
        # print("line: ", line, " vs spell checked line: ", spell_checked_line, " at time: ", time)
        spell_checked_lines.append((time, line))
    
    return spell_checked_lines

def text_to_captions(lines, delay: int = 0) -> WebVTT:
    vtt = WebVTT()
    for first_line, second_line in zip(lines, lines[1:]):
        beginning = format_timedelta(first_line[0] + timedelta(milliseconds=delay))
        end = format_timedelta(second_line[0] + timedelta(milliseconds=delay))
        caption = Caption(beginning, end, first_line[1])
        vtt.captions.append(caption)
        
    vtt.captions.append(Caption(format_timedelta(lines[-1][0] + timedelta(milliseconds=delay)), format_timedelta(lines[-1][0] + timedelta(seconds=5)), lines[-1][1]))
    return vtt

# Python arg parser instead of this bullshit
if __name__ == "__main__":
    video_file = sys.argv[1]
    song_name = sys.argv[2] if len(sys.argv) > 2 else video_file
    generate_new = len(sys.argv) > 3 and sys.argv[3] == "new"
    if not os.path.exists(f"{video_file}.pkl") or generate_new:
        frames = ccl_to_frames(video_file, f'./{song_name}')
        lines = frames_to_text(frames)
        pickle.dump(lines, open(f"{song_name}.pkl", "wb"))
    lines = pickle.load(open(f"{song_name}.pkl", "rb"))
    lines = remove_duplicate_lines(lines)
    lines = remove_invalid_lines(lines)
    lines = spell_check(lines)
    captions = text_to_captions(lines, 700)
    captions.save(f"{song_name}.vtt")

def video_file_to_captions(video_file: str) -> WebVTT:
    frames = ccl_to_frames(video_file, f'./{song_name}')
    lines = frames_to_text(frames)
    lines = remove_duplicate_lines(lines)
    lines = remove_invalid_lines(lines)
    captions = text_to_captions(lines, 750)
    return captions

#1:45 Every day the same old same old Time now to toss that rule Aespa spicy seems to start a couple seconds late??? FIx bug
# Same with 1:59