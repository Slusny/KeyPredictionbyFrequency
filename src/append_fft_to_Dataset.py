import pandas as pd
import os
import numpy as np
from src.fourier_analysis import getSpectum
from pathlib import Path
import matplotlib.pyplot as plt

target_folder = 'data/dl_more_piano_4000'
piano_2000_path  = 'data/dl_more_piano_4000'
file_name = 'dataset_abs.csv'
file_name_complex = 'dataset_complex.pkl'


def add_spectrum(file_path_string):
    head, tail = os.path.split(file_path_string)
    name = tail.split('.')[0]
    file_name = name + '.wav'
    file_path = Path.joinpath(Path.cwd(),piano_2000_path,file_name)#.parents[0]
    # file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..',piano_2000_path,file_name)
    if file_path.exists():
        spectrum, rate = getSpectum(file_path.as_posix(),name)
        return spectrum, rate
    else:
        print('path ',file_path.as_posix(),' does not exist')
        return np.nan, None


def create_new_dataset():
    df = pd.read_pickle(os.path.join(target_folder, "dataset.pkl"))
    spectrum_series = df['file_path'].apply(add_spectrum)
    rate = spectrum_series[0][1]
    length = len(spectrum_series[0][0])
    spectrum_series = spectrum_series.apply(lambda x: x[0])
    lengths = spectrum_series.apply(len)
    print(f'Deleting {len(lengths[lengths != length])} transforms with wrong length (should be {length})...')
    spectrum_series = spectrum_series[lengths == length]
    frqLabel = np.linspace(0.0, rate / 2.0, length)
    spectrum_df = pd.DataFrame.from_dict(dict(zip(spectrum_series.index, spectrum_series.values)), orient='index')
    spectrum_df.columns = frqLabel
    spectrum_df_abs = spectrum_df.apply(np.abs)
    joined_df = df.join(spectrum_df_abs, how='inner').reset_index(drop=True)
    joined_df_complex = df.join(spectrum_df, how='inner').reset_index(drop=True)
    print(joined_df.info())
    print(f'Saving dataframe to {os.path.join(target_folder, file_name)}')
    joined_df.to_csv(os.path.join(target_folder, file_name))
    joined_df_complex.to_pickle(os.path.join(target_folder, file_name_complex))


def test_new_dataset():
    df = pd.read_csv(os.path.join(target_folder, file_name))
    x = 'n'
    while x == 'n':
        row = df.sample()
        t_n = 30 #sec
        rate = 2000 
        plt.title(row['song_name'].values[0])
        spectrum = eval((row['spectrum'].values)[0])
        frqLabel = np.linspace(0.0, rate/2.0, len(spectrum))
        plt.plot(frqLabel, np.abs(spectrum))
        plt.show()
        print('type "n" for next sample, "s" for stop')
        x = input()

# to test a dataset    
#test_new_dataset()

## to create a dataset
create_new_dataset()