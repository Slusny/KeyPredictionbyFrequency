import pandas as pd
import os
import numpy as np
from fourier_analysis import getSpectum
from pathlib import Path
import matplotlib.pyplot as plt

target_folder = 'data/piano'
piano_2000_path  = 'data/piano_2000'
file_name = 'dataset_with_spectrum.csv'

def add_spectrum(row):
    head, tail = os.path.split(row[index])
    name = tail.split('.')[0]
    file_name = name + '.wav'
    file_path = Path.joinpath(Path.cwd(),piano_2000_path,file_name)#.parents[0]
    # file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..',piano_2000_path,file_name)
    if file_path.exists():
        spectrum, rate = getSpectum(file_path.as_posix(),name)
        return spectrum.tolist()
    else:
        print('path ',file_path.as_posix(),' does not exist')
        return np.nan

def create_new_dataset():
    df = pd.read_pickle(os.path.join(target_folder, "dataset.pkl"))
    index = np.where(df.columns.to_numpy() == 'file_path')[0][0]
    df['spectrum'] = df.apply(lambda r: tuple(r), axis=1).apply(add_spectrum)

    df.to_csv(os.path.join(target_folder, file_name))
    
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
test_new_dataset()

## to create a dataset
#create_new_dataset