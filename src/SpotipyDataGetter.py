import spotipy
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyOAuth
from src import credentials
import pandas as pd
import urllib.request
import os
import time


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class SpotipyDataGetter():
    def __init__(self):
        self.spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=credentials.client_id,
                                                                 client_secret=credentials.client_secret,
                                                                 redirect_uri=credentials.redirect_uri))


    def download_clips(self, url_list, target_folder):
        '''
        Downloads 30 second previews of songs via the Spotify API.
        See https://developer.spotify.com/documentation/web-api/reference/#/operations/get-track for details.

        :param url_list: A list-like object containing the spotify URLs of the songs to be downloaded
        :param target_folder: The target folder
        :return: A dataframe containing the requested URLs, the name of the song, the popularity
                 of the song as measured by spotify, and the location of the downloaded file on disk.
        '''
        num_fails = 0
        print('Downloading preview clips...')

        # spotipy allows only getting 50 tracks per API call
        df_list = []
        for i, url_chunk in enumerate(chunks(url_list, 50)):
            print(f'Clip chunk: {i}')
            while True:
                try:
                    tracks = self.spotify.tracks(tracks=url_chunk)
                except SpotifyException():
                    print(f'Could not retrieve clip URL for chunk {i}. Waiting for 30s...')
                    time.sleep(30)
                    continue
                break

            dic_list = []
            for i, tr in enumerate(tracks['tracks']):
                if tr['preview_url'] is None:
                    num_fails = num_fails +1
                    save_path = None
                else:
                    save_path = os.path.join(target_folder, f"{tr['uri'].split(':')[-1]}.mp3").replace('\\', '/')
                    urllib.request.urlretrieve(tr['preview_url'], save_path)
                dic = {'URL': url_chunk[i], 'song_name': tr['name'], 'popularity': tr['popularity'], 'file_path': save_path}
                dic_list.append(dic)
            df_list.append(pd.DataFrame(dic_list))

        print(f'Successfully downloaded {len(url_list) - num_fails} songs.')
        print(f'Failed to download {num_fails} songs.')
        return pd.concat(df_list, ignore_index=True)


    def get_audio_features(self, url_list):
        '''
        Gets audio features of songs via the Spotify API

        :param url_list: A list-like object containing the spotify URLs of the songs
        :return: A pandas dataframe containing the URLs and the returned audio features as columns
        '''
        print('Getting audio features...')
        df_list = []
        for i, url_chunk in enumerate(chunks(url_list, 100)):
            print(f'Audio features chunk: {i}')
            while True:
                try:
                    audio_features = self.spotify.audio_features(tracks=url_chunk)
                except SpotifyException():
                    print(f'Could not retrieve audio features for chunk {i}. Waiting for 30s...')
                    time.sleep(30)
                    continue
                break
            df = pd.DataFrame(audio_features)
            df.insert(0, 'URL', url_chunk)
            df_list.append(df)
        return pd.concat(df_list, ignore_index=True)


    def get_audio_analysis(self, url_list, features=['key', 'key_confidence', 'mode', 'mode_confidence', 'tempo', 'tempo_confidence']):
        '''
        Gets audio analysis of songs via the Spotify API.
        At the moment, this returns only selected elements from the "track" section.
        See https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-analysis for details.

        :param url_list: A list-like object containing the spotify URLs of the songs
        :return: A pandas dataframe containing the URLs and the returned audio features as columns

        '''
        print('Getting audio analysis...')
        analysis_list = []
        for i, url in enumerate(url_list):
            print(f'Audio analysis: {i}')
            while True:
                try:
                    audio_ana = self.spotify.audio_analysis(track_id=url)
                except SpotifyException():
                    print(f'Could not retrieve audio analysis for song {url}. Waiting for 30s...')
                    time.sleep(30)
                    continue
                break
            analysis_list.append(audio_ana['track'])
        df = pd.DataFrame(analysis_list)[features]
        df.insert(0, 'URL', url_list)
        return df


    def get_full_dataset(self, url_list, target_folder, file_name='dataset.pkl'):
        '''
        Gets the full dataset including labels, audio features, and a column "file_path" that points to the downloaded audio clips.
        Also saves the dataset as pickle file.

        :param url_list: A list-like object containing the spotify URLs of the songs. Must not contain duplicates.
        :param target_folder: The target folder for saving the mp3 clips and dataframe
        :param file_name: File name for the pkl file to store the dataframe
        :return: The desired dataframe
        '''
        print('Getting full dataset:')
        os.makedirs(target_folder, exist_ok=True)
        df_ana = self.get_audio_analysis(url_list)
        df_feat = self.get_audio_features(url_list)
        df_clips = self.download_clips(url_list, target_folder)
        cols_to_delete = df_feat.columns[df_feat.columns.isin(df_ana.columns)]
        cols_to_delete = cols_to_delete[cols_to_delete != 'URL']
        df_feat = df_feat.drop(columns=cols_to_delete)  # features from the analysis endpoint are preferred
        merged = pd.merge(df_ana, df_feat, on='URL')
        merged = pd.merge(merged, df_clips, on='URL')
        merged = merged.drop_duplicates().dropna()  # only clean and complete data
        print(f'Final dataset size: {merged.shape}')
        merged.to_pickle(os.path.join(target_folder, file_name))
        return merged


    def get_playlist_tracks(self, playlist_url):
        '''

        :param playlist_url: The url of the playlist
        :return: A list of the urls of all tracks in the playlist
        '''
        print('Getting tracks...')
        ls = []
        results = self.spotify.playlist_items(playlist_url, fields='next,items.track.external_urls.spotify', limit=50)
        ls.extend([el['track']['external_urls']['spotify'] for el in results['items']])
        while results['next']:
            results = self.spotify.next(results)
            ls.extend([el['track']['external_urls']['spotify'] for el in results['items']])

        ls = list(filter(lambda x: x is not None, ls))
        return ls


    def get_dataset_from_playlist(self, playlist_url, target_folder, file_name='dataset.pkl'):
        '''
        :param playlist_url: The url of the playlist
        :param target_folder: The target folder for audio files and dataset.pkl
        :param file_name: File name for the pkl file to store the dataframe
        :return: The desired dataset
        '''
        tracks = self.get_playlist_tracks(playlist_url)
        return self.get_full_dataset(tracks, target_folder=target_folder, file_name=file_name)




if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 200)

    url_list = ['https://open.spotify.com/track/4aWmUDTfIPGksMNLV2rQP2', 'https://open.spotify.com/track/7qiZfU4dY1lWllzX7mPBI3']
    sd = SpotipyDataGetter()

    #dataset = sd.get_full_dataset(url_list, 'data')
    #print(dataset.info())

    piano_playlist_url = 'https://open.spotify.com/playlist/4XJoQM1WZeJWqpV7iKszHM'
    test_url = 'https://open.spotify.com/playlist/4295JDoKFcnzqjwnyFSUgW'
    val = sd.get_dataset_from_playlist(piano_playlist_url, '../data/piano')
