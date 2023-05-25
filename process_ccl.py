import difflib
import itertools
import os
import pickle
import re
import sys
from datetime import timedelta

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
    
    # Hardcoded Crop... I should figure out a way to detect cropped boundaries
    frame_x_begin = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.8)
    frame_x_end = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.967)
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
        curr_time = timedelta(seconds=count / fps)
        count += 1
        
        
        is_prev_curr_and_different = is_different_frame(previous_frame, cropped)
        
        # Don't know why but pretransition frame always needs to be before cropped???
        is_prev_trans_and_curr_different = is_different_frame(pre_transition_frame, cropped)
        is_prev_frame_too_soon = curr_time - unique_frames[-1][0] > timedelta(milliseconds=500) if len(unique_frames) > 0 else True
        if is_prev_curr_and_different and not transitioning:
            transitioning = True
            pre_transition_frame = previous_frame
            # print("Found transition start with mse:", mse_between_frames(previous_frame, cropped), " at time", curr_time)
        elif not is_prev_curr_and_different and is_prev_trans_and_curr_different and transitioning and is_prev_frame_too_soon:
            transitioning = False
            # print("Found new unique frame with mse:", mse_between_frames(previous_frame, cropped), "at time", curr_time, "and success:", success)
            unique_frames.append((timedelta(seconds=count / fps), cropped))
            cv2.imwrite(os.path.join(dir, f"{format_timedelta(curr_time)}.jpg"), cropped)    
            success += 1
        
        previous_frame = cropped
    print("Found", len(unique_frames), "unique frames.")
    return unique_frames
        

def remove_ascii(text: str) -> str:
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

def frames_to_text(frames):
    print("Reading text from frames...")
    reader = Reader(['en', 'ko'], gpu=False)
    lines = []
    for frame in tqdm(frames):
        image = preprocess_image(frame[1])
        # cv2.imwrite(f"./recent/{frame[0]}.jpg", image)
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

def crop(image: np.ndarray, leeway: int = 10) -> np.ndarray:
    first_horizontal = 10000
    last_horizontal = 0
    first_vertical = 10000
    last_vertical = 0
    for x, pixels in enumerate(image):
        for y, pixel in enumerate(pixels):
            if pixel != 0:
                first_horizontal = min(first_horizontal, x) if first_horizontal is not None else x
                last_horizontal = max(last_horizontal, x) if last_horizontal is not None else x
                first_vertical = min(first_vertical, y) if first_vertical is not None else y
                last_vertical = max(last_vertical, y) if last_vertical is not None else y
    
    first_horizontal = max(0, first_horizontal - leeway)
    last_horizontal = min(image.shape[0], last_horizontal + leeway)
    first_vertical = max(0, first_vertical - leeway)
    last_vertical = min(image.shape[1], last_vertical + leeway)
    print(first_horizontal, last_horizontal, first_vertical, last_vertical)
    return image[first_horizontal:last_horizontal, first_vertical:last_vertical]

def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    image = crop(image)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    image = cv2.medianBlur(image,5)
    return image

# Kinda bad code design to only be able to pass timed_lines and not lines???
def remove_duplicate_lines(timed_lines):
    processed = []

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
    # engDict = enchant.Dict("en_US")
    
    valid_lines = []
    for time, line in timed_lines:
        valid_num_spaces = line.count(' ') > int(len(line) / 10)              
        # valid_num_eng_words = len([word for word in line.split() if engDict.check(word)]) > len(line.split()) / 4
        
        if valid_num_spaces:
            valid_lines.append((time, line))
    
    return valid_lines

def spell_check(timed_lines):
    spell_checked_lines = []
    for time, line in timed_lines:
        line = line.replace(';', '')
        
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
        
        for word in line.split():
            # Make sure word is properly capitalized
            word = word[0] + word[1:].lower()
        # sentences = re.findall('[A-Z][^A-Z]*', line)
        # spell_checked_line = ''.join([str(TextBlob(sentence).correct()) for sentence in sentences])
        # spell_checked_lines.append((time, spell_checked_line))
        # print("line: ", line, " vs spell checked line: ", spell_checked_line, " at time: ", time)
        spell_checked_lines.append((time, line))
    
    return spell_checked_lines

def text_to_captions(lines, lag: int = 700, delay: int = 0) -> WebVTT:
    vtt = WebVTT()
    for first_line, second_line in zip(lines, lines[1:]):
        beginning = format_timedelta(first_line[0] + timedelta(milliseconds=(delay + lag)))
        end = format_timedelta(second_line[0] + timedelta(milliseconds=(delay + lag)))
        caption = Caption(beginning, end, first_line[1])
        vtt.captions.append(caption)
        
    vtt.captions.append(Caption(format_timedelta(lines[-1][0] + timedelta(milliseconds=delay)), format_timedelta(lines[-1][0] + timedelta(seconds=5)), lines[-1][1]))
    return vtt



def video_file_to_captions(video_file: str) -> WebVTT:
    frames = ccl_to_frames(video_file, f'./{song_name}')
    lines = frames_to_text(frames)
    lines = remove_duplicate_lines(lines)
    # lines = remove_invalid_lines(lines)
    captions = text_to_captions(lines, 750)
    return captions

def get_youtube_link(search_term: str) -> str:
    prefix = 'https://www.youtube.com'
    
    results = YoutubeSearch(search_term, max_results=1).to_dict()[0]

    return prefix + results['url_suffix']

def download_videos(url: str, title: str, path: str):
    if not os.path.exists(path):
        os.mkdir(path)
        
    ydl_opts = {'paths': {'home': path}, 
                'outtmpl': f'{title}.%(ext)s',
                'writesubtitles': True,
                'subtitleslangs': ['en'],}

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download(url)

def download_subs_from_youtube(url: str, title: str, path: str) -> bool:
    if not os.path.exists(path):
        os.mkdir(path)
        
    ydl_opts = {'paths': {'home': path}, 
                'outtmpl': f'{title}.%(ext)s',
                'writesubtitles': True,
                'subtitleslangs': ['en-GB', 'en', 'en-US'],
                'skip_download': True,}
    
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download(url)
        
    # Shit code who cares; I do, cries :(
    if os.path.exists(f'{title}.en.vtt'):
        os.rename(f'{title}.en.vtt', f'{title}.vtt')
        return True
    elif os.path.exists(f'{title}.en-GB.vtt'):
        os.rename(f'{title}.en-GB.vtt', f'{title}.vtt')
        return True
    elif os.path.exists(f'{title}.en-US.vtt'):
        os.rename(f'{title}.en-US.vtt', f'{title}.vtt')
        return True
    else:
        return False
        
# Python arg parser instead of this bullshit
if __name__ == "__main__":
    video_file = sys.argv[1]
    song_name = sys.argv[2] if len(sys.argv) > 2 else video_file
    generate_new = len(sys.argv) > 3 and sys.argv[3] == "new"
    if not os.path.exists(f"./{song_name}/{song_name}.pkl") or generate_new:
        frames = ccl_to_frames(video_file, f'./{song_name}')
        lines = frames_to_text(frames)
        pickle.dump(lines, open(f"./{song_name}/{song_name}.pkl", "wb"))
    lines = pickle.load(open(f"./{song_name}/{song_name}.pkl", "rb"))
    lines = remove_duplicate_lines(lines)
    lines = spell_check(lines)
    pickle.dump(lines, open(f"./{song_name}/{song_name}_processed.pkl", "wb"))
    captions = text_to_captions(lines, delay = 0, lag = 750)
    captions.save(f"./subs/{song_name}.vtt")
    
    
def download_subs(artists: List[str], song_names: List[str], delays: List[int], sub_path: str, ccl_path: str):
    songs = [f'{s[0]} - {s[1]}' for s in list(zip(artists, song_names, delays))]
    ccl_songs = [song for song in songs if not download_subs_from_youtube(get_youtube_link(song[0], song[1]), song[1], sub_path)]
    
    for song in ccl_songs:
        download_videos(get_youtube_link(song[0], song[1]), song[1], ccl_path)
    
    file_names = [f"{song[0]} - {song[1]}" for song in ccl_songs]
    for file in file_names:
        if len(glob.glob(f"{ccl_path}/{file}*")) == 0:
            continue
        
        video_file = glob.glob(f"{ccl_path}/{file}*")[0]
        frames = ccl_to_frames(video_file, f'./{file}')
        lines = frames_to_text(frames)
        lines = remove_duplicate_lines(lines)
        lines = spell_check(lines)
        captions = text_to_captions(lines, delay = 0, lag = 750)
        captions.save(f"{path}/{file}.vtt")
    

        
    
#1:45 Every day the same old same old Time now to toss that rule Aespa spicy seems to start a couple seconds late??? FIx bug
# Same with 1:59