import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
from typing import List
import wikipedia
from bs4 import BeautifulSoup
import mysql.connector
import re
from dotenv import dotenv_values
import difflib
from typing import Tuple
from GenerateSubs import download_all_subs

scope = """ugc-image-upload,
user-read-recently-played, user-top-read, user-read-playback-position,
user-read-playback-state, user-modify-playback-state, user-read-currently-playing,
app-remote-control, streaming,
playlist-modify-public, playlist-modify-private,
playlist-read-private, playlist-read-collaborative,
user-follow-modify, user-follow-read,
user-library-modify, user-library-read,
user-read-email, user-read-private"""

## This API for my database is pretty hot trash... I should fix it up

class Artist:
    def __init__(self, artist_id: str, artist_name: str):
        self.artist_id = artist_id
        self.artist_name = artist_name
        
    def __str__(self):
        return f"{self.artist_name} ({self.artist_id})"
    
    def __repr__(self):
        return self.__str__()
    
class Song:
    def __init__(self, song_id: str, song_name: str):
        self.song_id = song_id
        self.song_name = song_name
        
    def __str__(self):
        return f"{self.song_name} ({self.song_id})"
    
    def __repr__(self):
        return self.__str__()

class KpopDatabase:
    def __init__(self, cnx, cursor):
        self.cnx = cnx
        self.cursor = cursor
        
    def close(self):
        self.cnx.close()
        self.cursor.close()
    
    def insert_artist(self, artist_id: str, artist_name: str):
        query = "SELECT * FROM Artists WHERE ArtistID = %s"
        self.cursor.execute(query, [artist_id])
        
        result = self.cursor.fetchone()
        if not result:
            insert_artist_query = "INSERT INTO Artists (ArtistID) VALUES (%s)"
            self.cursor.execute(insert_artist_query, [artist_id])
            self.cnx.commit()

        query = "SELECT * FROM ArtistNames WHERE ArtistID = %s AND ArtistName = %s"
        self.cursor.execute(query, [artist_id, artist_name])
        result = self.cursor.fetchone()
        if not result:
            insert_artist_name_query = "INSERT INTO ArtistNames (ArtistID, ArtistName) VALUES (%s, %s)"
            self.cursor.execute(insert_artist_name_query, [artist_id, artist_name])
            self.cnx.commit()
    
    def insert_song(self, song_id: str, song_name: str):
        query = "SELECT * FROM Songs WHERE SongID = %s"
        self.cursor.execute(query, [song_id])
        
        result = self.cursor.fetchone()
        if result:
            print(f"Song Name '{song_name}' already exists.")
        else:
            insert_song_query = "INSERT INTO Songs (SongID, SongName) VALUES (%s, %s)"
            arguments = [song_id, song_name]
            self.cursor.execute(insert_song_query, arguments)
            self.cnx.commit()    
    
    def insert_song_and_artist(self, artist_id: str, song_id: str):
        query = "SELECT * FROM SongArtists WHERE SongID = %s AND ArtistID = %s"
        self.cursor.execute(query, [song_id, artist_id])
        result = self.cursor.fetchone()
        
        if result:
            print(f"Song Artist '{artist_id}' already exists.")
        else:
            insert_song_artist_query = "INSERT INTO SongArtists (SongID, ArtistID) VALUES (%s, %s)"
            self.cursor.execute(insert_song_artist_query, [song_id, artist_id])
            self.cnx.commit()
    
    def insert_subtitles(self, song_id: str, subtitles: str):
        pass
    
    # This code doesn't check if an artist or song already exists in the database
    def remove_artist(self, artist_id: str):
        query = "DELETE FROM ArtistNames WHERE ArtistID = %s"
        cursor.execute(query, [artist_id])
        cnx.commit()
        
        query = "DELETE FROM SongArtists WHERE ArtistID = %s"
        cursor.execute(query, [artist_id])
        cnx.commit()
        
        query = "DELETE FROM Artists WHERE ArtistID = %s"
        cursor.execute(query, [artist_id])
        cnx.commit()
    
    def remove_song(self, song_id: str):
        query = "DELETE FROM SongArtists WHERE SongID = %s"
        self.cursor.execute(query, [song_id])
        self.cnx.commit()
        
        query = "DELETE FROM Songs WHERE SongID = %s"
        self.cursor.execute(query, [song_id])
        self.cnx.commit()
    
    def remove_song_and_artist(self, artist_id: str, song_id: str):
        query = "DELETE FROM SongArtists WHERE SongID = %s AND ArtistID = %s"
        self.cursor.execute(query, [song_id, artist_id])
        self.cnx.commit()
    
    def remove_subtitles(self, song_id: str):
        query = "DELETE FROM Subtitles WHERE SongID = %s"
        self.cursor.execute(query, [song_id])
        self.cnx.commit()
    
    # SQL INJECTION TIME
    def get_artists(self, artist_ids: List[str]) -> List[str]:
        if len(artist_ids) == 0:
            return []
        
        query = "SELECT ArtistName FROM ArtistNames WHERE ArtistID IN (%s)"
        placeholders = ', '.join(['%s'] * len(artist_ids))
        query = query % placeholders
        
        self.cursor.execute(query, artist_ids)
        return [artist_name[0] for artist_name in self.cursor.fetchall()]
    
    def get_songs(self, song_ids: List[str]) -> List[Song]:
        if len(song_ids) == 0:
            return []
        
        query = "SELECT SongName, SongID FROM Songs WHERE SongID IN (%s)"
        placeholders = ', '.join(['%s'] * len(song_ids))
        query = query % placeholders

        self.cursor.execute(query, song_ids)
        return [Song(song_name[1], song_name[0]) for song_name in self.cursor.fetchall()]
    
    def get_subtitles(self, song_ids: List[str]) -> List[str]:
        if len(song_ids) == 0:
            return []

        query = "SELECT Subtitles FROM Subtitles WHERE SongID IN (%s)"
        placeholders = ', '.join(['%s'] * len(song_ids))
        
    
    def get_songs_artists(self, song_id: str) -> List[Artist]:
        query = "SELECT SongArtists.ArtistID, ArtistName FROM SongArtists INNER Join ArtistNames on ArtistNames.ArtistID = SongArtists.ArtistId WHERE SongID = %s"
        self.cursor.execute(query, [song_id])
        return [Artist(artist_id[0], artist_id[1]) for artist_id in self.cursor.fetchall()]
    
    def get_artists_songs(self, artist_id: str) -> List[Song]:
        query = "SELECT SongID, SongName FROM SongArtists WHERE ArtistID = %s"
        self.cursor.execute(query, [artist_id])
        return [Song(song_id[0], song_id[1]) for song_id in self.cursor.fetchall()]
    
    def find_mentioned_artists(self, search_string: str) -> List[Artist]:
        # Prepare the query with a placeholder for the search string
        query = "SELECT DISTINCT ArtistID, ArtistName FROM ArtistNames WHERE LOWER(%s) LIKE CONCAT('%', LOWER(ArtistName), '%')"

        # Execute the query with the search string as a parameter
        cursor.execute(query, [search_string.lower()])
        matching_artist_ids = cursor.fetchall()
        if len(matching_artist_ids) == 0:
            print(f"No matching artist names found for '{search_string}'")
            return []
            
        return list([Artist(artist_id[0], artist_id[1]) for artist_id in matching_artist_ids])

    def search_mentioned_songs(self, search_string: str) -> List[Song]:
        # Prepare the query with a placeholder for the search string
        query = "SELECT DISTINCT SongID, SongName FROM Songs WHERE LOWER(%s) LIKE CONCAT('%', LOWER(SongName), '%')"

        # Execute the query with the search string as a parameter
        self.cursor.execute(query, [search_string.lower()])
        matching_song_ids = self.cursor.fetchall()
        if len(matching_song_ids) == 0:
            print(f"No matching song names found for '{search_string}'")
            return []
            
        return list([Song(song_id[0], song_id[1]) for song_id in matching_song_ids])

    def search_mentioned_songs_with_artist_filter(self, search_string: str, artist_ids: List[str] = []) -> List[Song]:
        search_string = re.sub(r'[^a-zA-Z0-9\s]+', '', search_string)
        artist_ids = [a.artist_id for a in self.find_mentioned_artists(search_string)] if len(artist_ids) == 0 else artist_ids
        query = "SELECT Songs.SongID, Songs.SongName FROM Songs INNER JOIN SongArtists ON Songs.SongID = SongArtists.SongID WHERE LOWER(%s) LIKE CONCAT('%', LOWER(SongName), '%') AND ArtistID IN ({placeholders})"
        placeholders = ', '.join(['%s'] * len(artist_ids))
        query = query.format(placeholders=placeholders)
        params = [search_string] + [artist.lower() for artist in artist_ids]
        self.cursor.execute(query, params)

        matching_songs = self.cursor.fetchall()
        
        if len(matching_songs) == 0:
            print(f"No matching songs found for '{search_string}'")
            return []
            
        return list([Song(song[0], song[1]) for song in matching_songs])


def pretty_print(json_data: dict):
    print(json.dumps(json_data, sort_keys=True, indent=4))

def genereate_spotify_instance():
    # Don't judge me I don't want to worry about creating a new clientid...
    config = dotenv_values("/Users/alexchen/Coding/Projects/rspot/target/release/.env")
    SPOTIPY_CLIENT_ID = config['RSPOTIFY_CLIENT_ID']
    SPOTIPY_CLIENT_SECRET = config['RSPOTIFY_CLIENT_SECRET']
    SPOTIPY_REDIRECT_URI = config['RSPOTIFY_REDIRECT_URI']

    return spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope=scope, open_browser=True))

def find_artist(sp: spotipy.Spotify, artist_name: str, genre_filter: List[str] = []) -> Artist:
    genre_filter = " ".join([f"genre:{genre}" for genre in genre_filter])
    search_result = sp.search(f"artist:{artist_name} {genre_filter}", limit=5, type="artist")

    if len(search_result['artists']['items']) == 0:
        search_result = sp.search(f"artist:{artist_name}", limit=5, type="artist")
        if len(search_result['artists']['items']) == 0:
            return ("", "")
    
    correct_id = ''
    correct_name = ''
    diff = 100000
    for artist in search_result['artists']['items']:
        if artist['name'].lower() == artist_name.lower():
            correct_id = artist['id']
            correct_name = artist['name']
            break
        else:
            spaceless_found_artist_name = artist["name"].replace(' ', '').lower()
            spaceless_artist_name = artist_name.replace(' ', '').lower()
            output_list = [li for li in difflib.ndiff(spaceless_artist_name, spaceless_found_artist_name) if li[0] != ' ']
            if len(output_list) < diff:
                correct_id = artist['id']
                correct_name = artist['name']
                diff = len(output_list)
    
    print(f"Found artist '{correct_name}' from search '{artist_name}'")
    return Artist(correct_id, correct_name)

def get_kpop_groups() -> List[str]:
    articles = ["List of South Korean idol groups (2020s)", "List of South Korean idol groups (2010s)", "List of South Korean idol groups (2000s)"]
    artists = []    
    for article in articles:
        result = wikipedia.page(article).html()
        soup = BeautifulSoup(result, "html.parser")
        dec_artists = [year.find_all("li") for year in soup.find_all("div", class_="div-col")]
        dec_artists = sum(dec_artists, [])
        dec_artists = [artist.text for artist in dec_artists]
        artists += dec_artists
        
    return artists

def find_artists_songs(sp, artist_name) -> List[dict]:
    albums = sp.artist_albums(artist_name, limit=50)
    tracks = []
    for album in albums["items"]:
        tracks += sp.album_tracks(album["id"])['items']
     
    return tracks
    
def insert_all_artists_and_songs(kpopDatabase: KpopDatabase, artist_names: List[str]):
    banned_artist_ids = ["2019zR22qK2RBvCqtudBaI", "1ZwdS5xdxEREPySFridCfh"]
    banned_artist_names = ['Homme', 'One Way', 'Touch', 'AA', 'C-REAL', 'N-Sonic', 'Skarf', 'The East Light', 'T-ara N4']
    failed_artists = []
    for artist_name in artist_names:
        if artist in banned_artist_names:
            continue
        sp_artist = find_artist(sp, artist_name, ["k-pop"])
        artist_id = sp_artist.artist_id
        artist_name_sp = sp_artist.artist_name

        if artist_id == "" or artist_id in banned_artist_ids:
            print(f"Could not find artist {artist_name}")
            failed_artists.append(artist_name)
            continue
            
            
        kpopDatabase.insert_artist(artist_id, artist_name)
        kpopDatabase.insert_artist(artist_id, artist_name_sp)
        
        tracks = find_artists_songs(sp, artist_id)
        for track in tracks:
            # print("Inserting song: " + track[1])
            kpopDatabase.insert_song(track['id'], track['name'])
            kpopDatabase.insert_song_and_artist(artist_id, track['id'])
            
    
    for artist_name in failed_artists:
        print(artist_name)
        
        

# Connect to the MySQL database
cnx = mysql.connector.connect(
    user="root",
    database="kpop_songs"
)

cursor = cnx.cursor()
sp = genereate_spotify_instance()

kpopDatabase = KpopDatabase(cnx, cursor)
# artists = ["Pristin V", "Astro – Moonbin & Sanha", "E'Last", "Astro – Jinjin & Rocky", "Hello Venus", "NU'EST", "NU'EST W"]
# artists = get_kpop_groups()
# insert_all_artists_and_songs(sp, cnx, cursor, artists)

search_string = "[BE ORIGINAL] NCT 127 'Sticker' (4K)"
ids = [id.song_id for id in kpopDatabase.search_mentioned_songs_with_artist_filter(search_string)]
print(kpopDatabase.get_songs(ids), [kpopDatabase.get_songs_artists(id) for id in ids])
kpopDatabase.close()


def format_song_and_artist(song_name: str, artist_name: str):
    return f"{artist_name} - {song_name}"


# This function is bad design? I'd rather split up some of this functionality
def get_subtitles(kpopDatabase: KpopDatabase, url: str) -> List[str]:
    search_string = some_function_that_gets_details(url)
    songs = kpopDatabase.search_mentioned_songs_with_artist_filter(search_string)
    if len(songs) == 0:
        return "nobody found or something"
        
    subtitles = kpopDatabase.get_subtitles(songs)
    # If no subtitles found, ask the user to choose which song they want 
    song = None
    artists = kpopDatabase.get_songs_artists(song)
    
    if len(artists) > 0:
        artist_name = artists[0].artist_name
    else:
        return "no artists found"
        # This should technically never be triggered?
        
    if len(subtitles) == 0:
        download_all_subs(song.song_name, artist_name, format_song_and_artist(song.song_name, artist_name))
        


