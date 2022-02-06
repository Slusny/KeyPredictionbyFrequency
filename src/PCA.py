from time import time
import numpy as np
import pandas as pd
import os 
from fourier_analysis import getSpectum
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
import time
import argparse
from pathlib import Path
from numpy.fft import irfft

def PCA(data_path,save_path):
    df = pd.read_pickle(data_path)
    spectrum_matix = (df.iloc[:,25:]).to_numpy()
    m = np.mean(spectrum_matix,axis=0)
    spectrum_matix -= m
    print("start svd")
    start_time = time.time()
    Q, S, V = np.linalg.svd(spectrum_matix)
    print("end svd, took : %i sec" %(time.time()-start_time))
    np.save(os.path.join(save_path,'SingularValues.npy'),S)
    np.save(os.path.join(save_path,'PrincipaleComponents.npy'),V)
    #plot
    x = np.arange(0,S.size)
    plt.title("Singular Values")
    plt.plot(x,S**2)
    plt.show()

def transform_song(save_path_songs,save_path,order,spectrum,name):
    V = np.load(os.path.join(save_path,'PrincipaleComponents.npy'))
    rate = (V.shape[0]-1)//15 # Datapoints / 2 / song length (30s)
    reconstructed = ((spectrum @ V.conj().T[:,:order]) @ V[:order,:])
    sound_back = irfft(reconstructed)
    wav.write(os.path.join(save_path_songs,name + '_PC_'+str(order)+'.wav'), rate, np.int16(sound_back* 32767 / sound_back.max()).squeeze())

def print_PC(save_path_songs,save_path,order):
    V = np.load(os.path.join(save_path,'PrincipaleComponents.npy'))
    rate = (V.shape[0]-1)//15 # Datapoints / 2 / song length (30s)
    # plot
    x = np.arange(0,V.shape[0])
    plt.title("Singular Values")
    plt.plot(x,np.abs(V[order]))
    plt.show()
    # save wav file
    sound_back = irfft(V[order])
    wav.write(os.path.join(save_path_songs,'PC_'+str(order)+'.wav'), rate, np.int16(sound_back* 32767 / sound_back.max()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--PCA', action='store_true', help="calculate PCA and save Singular Values and PC's")
    parser.add_argument('--PC_Order', type=int, default=30001, help='type in the order of the principal component you want plotted and an audio file saved of')
    parser.add_argument('--songName', type=str, help='type in the name of the song you want transformed. It will check the dataframe if any song names contain this string. If left empty a random song is transformed')
    parser.add_argument('--printPC', action='store_true', default=False, help='Show the Principal Component of order --PC_Order')
    parser.add_argument('--save_path', type=str, default='data', help='path where to save Principal Component and Singular Value matrix (.npy)')
    parser.add_argument('--save_path_songs', type=str, default='data', help='path where to save the transformed songs and PCs')
    parser.add_argument('--data_path', type=str, default='data/piano/dataset_spectrum.pkl', help='path to the pickle dataframe with spectrum')
    

    args = parser.parse_args()
    save_path_songs = args.save_path_songs
    data_path = args.data_path
    save_path = args.save_path

    if(args.PCA):
        PCA(data_path,save_path)
    else:
        PCpath = Path.joinpath(Path.cwd(),save_path,'PrincipaleComponents.npy') 
        if not PCpath.exists():
            print("can not find " + PCpath.as_posix() + ", do --PCA first")
        else:
            if args.printPC:
                print_PC(args.save_path_songs,args.save_path,args.PC_Order)
            else:
                df = pd.read_pickle(data_path)
                if not args.songName:
                    row = df.sample()
                    spectrum = row.iloc[:,25:].to_numpy()
                    name = row['song_name'].values[0]
                    print('transforming '+ name)
                else:
                    row = df[df['song_name'].str.contains(args.songName, case=False)]
                    if(row.shape[0] == 0):
                        print("no song with this name was found")
                    else:
                        print('found:')
                        print('    '+ '\n    '.join(row['song_name'].tolist())+'\nselecting the first one')
                        name = row.iloc[0]['song_name']
                        spectrum = row.iloc[0,25:].to_numpy()[None]
            
                transform_song(args.save_path_songs,args.save_path,args.PC_Order,spectrum,name)
