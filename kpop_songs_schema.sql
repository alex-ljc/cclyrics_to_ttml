CREATE DATABASE IF NOT EXISTS kpop_songs;

-- Set the newly created database as the default database
USE kpop_songs;

-- Create table Artists if it does not exist
CREATE TABLE IF NOT EXISTS Artists (
    ArtistID VARCHAR(50) PRIMARY KEY
);

-- Create table ArtistNames if it does not exist
CREATE TABLE IF NOT EXISTS ArtistNames (
    ArtistNameID INT AUTO_INCREMENT PRIMARY KEY,
    ArtistID VARCHAR(50),
    ArtistName VARCHAR(100),
    FOREIGN KEY (ArtistID) REFERENCES Artists(ArtistID)
);

-- Create table Songs if it does not exist
CREATE TABLE IF NOT EXISTS Songs (
    SongID VARCHAR(50) PRIMARY KEY,
    SongName VARCHAR(200),
);

-- Create table SongArtists if it does not exist
CREATE TABLE IF NOT EXISTS SongArtists (
    SongID VARCHAR(50),
    ArtistID VARCHAR(50),
    FOREIGN KEY (SongID) REFERENCES Songs(SongID),
    FOREIGN KEY (ArtistID) REFERENCES Artists(ArtistID)
);

CREATE TABLE IF NOT EXISTS Subtitles (
    SubtitleID INT AUTO_INCREMENT PRIMARY KEY,
    SongID VARCHAR(50),
    Subtitle VARCHAR(200) DEFAULT NULL,
    FOREIGN KEY (SongID) REFERENCES Songs(SongID)
)