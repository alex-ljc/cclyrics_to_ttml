import os
import pickle
import re
import sys
from datetime import timedelta

from lyricsgenius import Genius
from webvtt import Caption, WebVTT
from youtube_search import YoutubeSearch
from yt_dlp import YoutubeDL
from typing import List
import glob
import ProcessCCL

    
def search_for_video(search_term: str, key_word: str) -> str:
    search_terms = search_term + ' ' + key_word
    prefix = 'https://www.youtube.com'
    
    results = YoutubeSearch(search_terms, max_results=1).to_dict()[0]
    
    return prefix + results['url_suffix']

def download_video(url: str, title: str, path: str):
    if not os.path.exists(path):
        os.mkdir(path)
        
    ydl_opts = {'paths': {'home': path}, 
                'outtmpl': f'{title}.%(ext)s',
    }

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
    sub_path = f'{path}/{title}'
    if os.path.exists(f'{sub_path}.en.vtt'):
        os.rename(f'{sub_path}.en.vtt', f'{sub_path}.vtt')
        vtt_to_srt(f'{sub_path}.vtt', f'{sub_path}.srt')
        os.remove(f'{sub_path}.vtt')
        return True
    elif os.path.exists(f'{sub_path}.en-GB.vtt'):
        os.rename(f'{sub_path}.en-GB.vtt', f'{sub_path}.vtt')
        vtt_to_srt(f'{sub_path}.vtt', f'{sub_path}.srt')
        os.remove(f'{sub_path}.vtt')
        return True
    elif os.path.exists(f'{sub_path}.en-US.vtt'):
        os.rename(f'{sub_path}.en-US.vtt', f'{sub_path}.vtt')
        vtt_to_srt(f'{sub_path}.vtt', f'{sub_path}.srt')
        os.remove(f'{sub_path}.vtt')
        return True
    else:
        return False

# Pass through methods. These should be stored in a config file somewhere instead        
def download_ccl_subs(song_name: str, artist_name: str, sub_name: str, sub_path: str = '/Users/alexchen/Movies/subs', ccl_path: str = '/Users/alexchen/Movies/subs/ccl', use_cache: bool = True) -> str: 
    search_term = f"{artist_name} - {song_name}"
    # Probably want some error handling here in case we pass in bad inputs
    download_video(search_for_video(search_term, "colour coded lyrics"), search_term, ccl_path)
    
    # This assumes that download videos downloads to a specific location with file name equal to the search_term
    if len(glob.glob(f"{ccl_path}/{search_term}*")) == 0:
        return ""
    # Should raise some sort of exceptiion here
    
    video_file = glob.glob(f"{ccl_path}/{search_term}*")[0]
    captions = ProcessCCL.video_file_to_text(video_file, search_term, 750, use_cache=use_cache)
    captions.save_as_srt(f"{sub_path}/{sub_name}.srt")
    return f"{sub_path}/{sub_name}.srt"
    
    
def vtt_to_srt(vtt_file: str, srt_file_path):
    print(vtt_file)
    captions = WebVTT().read(vtt_file)
    captions.save_as_srt(srt_file_path)


    
