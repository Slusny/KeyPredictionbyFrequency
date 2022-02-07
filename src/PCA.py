from time import time
from turtle import color
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

rate = 2000

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
    np.save(os.path.join(save_path,'PrincipalComponents.npy'),V)
    #plot
    # x = np.linspace(0,rate/2,S.size)
    # plt.title("Singular Values")
    # plt.plot(x,S**2)
    # plt.show()

def transform_song(save_path_songs,save_path,order,spectrum,name,plot=False):
    V = np.load(os.path.join(save_path,'PrincipalComponents.npy'))
    rate = (V.shape[0]-1)//15 # Datapoints / 2 / song length (30s)
    reconstructed = ((spectrum @ V.conj().T[:,:order]) @ V[:order,:])
    sound_back = irfft(reconstructed)
    wav.write(os.path.join(save_path_songs,name + '_PC_'+str(order)+'.wav'), rate, np.int16(sound_back* 32767 / sound_back.max()).squeeze())
    if plot:
        return reconstructed

def print_PC(save_path_songs,save_path,order):
    V = np.load(os.path.join(save_path,'PrincipalComponents.npy'))
    rate = (V.shape[0]-1)//15 # Datapoints / 2 / song length (30s)
    # plot
    x = np.linspace(0,rate//2,V.shape[0])
    fig = plt.figure()
    plt.title("Principal Component "+str(order))
    plt.plot(x,np.abs(V[order-1]))
    plt.xlabel('Hz')
    # plt.show()
    plt.savefig(os.path.join(save_path,'PC_'+ str(order) +'.pdf'))
    # save wav file
    sound_back = irfft(V[order-1])
    wav.write(os.path.join(save_path_songs,'PC_'+str(order)+'.wav'), rate, np.int16(sound_back* 32767 / sound_back.max()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate the complex principal component of a fourier dataset and display the results as audio files and plot')
    parser.add_argument('--PCA', action='store_true', help="calculate PCA and save Singular Values and PC's")
    parser.add_argument('--PC_Order', type=int, default=30001, help='type in the order of the principal component you want plotted and an audio file saved of')
    parser.add_argument('--songName', type=str, help='type in the name of the song you want transformed. It will check the dataframe if any song names contain this string. If left empty a random song is transformed')
    parser.add_argument('--printPC', action='store_true', default=False, help='Show the Principal Component of order --PC_Order')
    parser.add_argument('--save_path', type=str, default='data', help='path where to save Principal Component and Singular Value matrix (.npy)')
    parser.add_argument('--save_path_songs', type=str, default='data', help='path where to save the transformed songs and PCs')
    parser.add_argument('--data_path', type=str, default='data/piano/dataset_complex.pkl', help='path to the pickle dataframe with spectrum')
    parser.add_argument('--plot', action='store_true', default=False)
    

    args = parser.parse_args()
    save_path_songs = args.save_path_songs
    data_path = args.data_path
    save_path = args.save_path

    if(args.PCA):
        PCA(data_path,save_path)
    else:
        PCpath = Path.joinpath(Path.cwd(),save_path,'PrincipalComponents.npy') 
        if not PCpath.exists():
            print("can not find " + PCpath.as_posix() + ", do --PCA first")
        else:
            if args.printPC:
                if(args.plot):
                    orders = [1,2,3,4,5,8,10,20,50,100,200,300,400,500,600,700,800,950]
                    for i,order in enumerate(orders):
                        print(str(i) + " of " + str(len(orders)) + ", order:" + str(order))
                        print_PC(args.save_path_songs,args.save_path,order)
                else:
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
                if(args.plot):
                    if(False):
                        orders = [1,2,3,4,5,8,10,20,50,100,200,300,400,500,600,700,800,950]
                        for i,order in enumerate(orders):
                            print(str(i) + " of " + str(len(orders)) + ", order:" + str(order))
                            reconstructed = transform_song(args.save_path_songs,args.save_path,order,spectrum,name,True)
                            reconstructed = reconstructed.squeeze()
                            fig = plt.figure()
                            plt.title(name + "projected onto "+str(order) + "Principal Components")
                            x = np.linspace(0,rate//2.,reconstructed.size)
                            plt.plot(x,np.abs(reconstructed))
                            plt.xlabel('Hz')
                            # plt.show()
                            plt.savefig(os.path.join(save_path,name+'_PC_'+ str(order) +'.pdf'))
                    else:
                        spectri = []
                        orders=[5,100,500,950][::-1]
                        fsize= 20
                        colors1 = ['#1B068C','#8204A7','#B22C8E','#EC7754']
                        colors = ['#18068B','#8F0DA3','#EB7654','#FDBE29'][::-1] 
                        # colors = ['#18068B','#8F0DA3','#EB7654','#F7D13C'][::-1]
                        for i,order in enumerate(orders):
                            print(str(i+1) + " of "+str(len(orders)) + " - order:" +str(order))
                            spectri.append(transform_song(args.save_path_songs,args.save_path,order,spectrum,name,True))
                        x = np.linspace(0,rate//2,spectri[0].size)
                        fig, ax1 = plt.subplots()
                        fig.set_figheight(7)
                        fig.set_figwidth(14)
                        plt.title("Reconstruction of "+name,fontsize=fsize)
                        S2 = np.load(os.path.join(save_path,'SingularValues.npy'))**2
                        xx = np.linspace(0,rate/2,S2.size)
                        ax1.plot(xx,S2,linewidth=3,label="eigenvalues")
                        ax2 = ax1.twinx()
                        ax1.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
                        ax1.patch.set_visible(False) # hide the 'canvas'
                        ax1.set_xlabel('Hz & Number-PC',fontsize=fsize)
                        ax2.set_ylabel('Amplitude of Frequency',fontsize=fsize)
                        ax1.set_ylabel('Eigenvalue size',fontsize=fsize)
                        for i,spectrum in enumerate(spectri):
                            # ax2.plot(x,np.clip(np.abs(spectrum.squeeze()[:S2.size]),0,200000),alpha=0.2,label="Order "+str(orders[i]),color=colors[i])
                            T = np.abs(spectrum.squeeze()).astype(np.double)
                            ax2.fill_between(x,T,label="Order "+str(orders[i]),color=colors[i])
                            # ax2.fill_between(x,np.clip(T,0,8e6),label="Order "+str(orders[i]),color=colors[i])
                            # ax2.plot(x,np.clip(T,0,8e6),color=colors[i])
                            # ax2.plot(x,T,color=colors[i])
                            ax1.scatter(orders[i],S2[orders[i]],color=colors[i],s=250,zorder=10,label="PC "+str(orders[i]))
                        ax1.legend(fontsize=fsize)
                        # plt.xticks(fontsize=fsize)
                        # plt.yticks(fontsize=fsize)
                        ax1.tick_params(axis='x', labelsize=fsize)

                        ax1.tick_params(axis='y', labelsize=fsize)
                        ax2.tick_params(axis='y', labelsize=fsize)
                        ax1.yaxis.offsetText.set_fontsize(fsize)
                        ax2.yaxis.offsetText.set_fontsize(fsize)
                        plt.tight_layout()
                        # fig2 = plt.figure(2)
                        # plt.title("original spectrum")
                        # plt.plot(x,np.abs(spectrum.squeeze()))
                        # plt.show()
                        plt.savefig(os.path.join(save_path,'PCA.pdf'))
                else:
                    transform_song(args.save_path_songs,args.save_path,args.PC_Order,spectrum,name)
