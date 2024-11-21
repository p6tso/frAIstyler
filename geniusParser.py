import lyricsgenius
from secret import GENIUS_TOKEN

genius = lyricsgenius.Genius(GENIUS_TOKEN)


def make_lyrics_base(name: str, num_songs: int = 20):
    artist = genius.search_artist(name, max_songs=num_songs, sort="title")
    with open(f'{name}.txt', 'w') as f:
        for song in artist.songs:
            lyrics = artist.song(song.title).lyrics
            f.write(lyrics + '\n')



def text_edit(file: str) -> list:
    with open(file) as f:
        data = f.read().split('Embed')
    edited_text = []
    for song in data:
        song = song.split('\n')
        edited_song = ''
        for line in song:
            if line.count('[') == 0:
                edited_song += f'{line}\n'
        edited_text.append(edited_song)
    return edited_text


artist_list = ['AQUAKEY', 'KUNTEYNIR', 'Ежемесячные']

num = 100

#for artist in artist_list:
#    make_lyrics_base(artist, num)

for artist in artist_list:
    edited = text_edit(f'{artist}.txt')
    with open(f'{artist}_edited.txt', 'w') as f:
        f.write('\n'.join(edited))
