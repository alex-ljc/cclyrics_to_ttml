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

class Song:
    def __init__(self, artist: str, song_name: str):
        self.song_name = song_name
        self.artist = artist
        self.search_term = self.get_search_term()
        self.youtube_link = self.get_youtube_link()
        
    def __str__(self):
        return f"{self.artist} - {self.song_name}"
    
    def get_search_term(self) -> str:
        return self.__str__()
        
    def get_youtube_link(self) -> str:
        search_terms = self.search_term + ' color coded lyrics'
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
        
def download_subs(song: Song, sub_name: str, sub_path: str = '/Users/alexchen/Movies/subs', ccl_path: str = '/Users/alexchen/Movies/subs/ccl', use_cache: bool = True): 
    if download_subs_from_youtube(song.youtube_link, sub_name, sub_path):
        return
    # Probably want some error handling here in case we pass in bad inputs

    download_video(song.youtube_link, song.search_term, ccl_path)
    
    # This assumes that download videos downloads to a specific location with file name equal to the search_term
    if len(glob.glob(f"{ccl_path}/{song.search_term}*")) == 0:
        return
    # Should raise some sort of exceptiion here
    
    video_file = glob.glob(f"{ccl_path}/{song.search_term}*")[0]
    captions = ProcessCCL.video_file_to_text(video_file, song.search_term, 750, use_cache=use_cache)
    captions.save_as_srt(f"{sub_path}/{sub_name}.srt")
    
def vtt_to_srt(vtt_file: str, srt_file_path):
    print(vtt_file)
    captions = WebVTT().read(vtt_file)
    captions.save_as_srt(srt_file_path)
    

def create_synced_video(url: str, artist: str, song_name: str, video_name: str, path: str):
    download_video(url, video_name, path)
    download_subs(Song(artist, song_name), video_name, sub_path = path, use_cache=False)
    

song1 = Song('BTS', 'Dynamite')
song2 = Song('TXT', 'Blue Hour')
url = sys.argv[1]
artist = sys.argv[2]
song_name = sys.argv[3]
title = sys.argv[4]
create_synced_video(url, artist, song_name, title, '/Users/alexchen/Movies/Performances')