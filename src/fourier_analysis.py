import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft, ifft
import numpy as np
import subprocess
import os
import math

def getDivisors(n, res=None) : 
    res = res or []
    i = 1
    while i <= n : 
        if (n % i==0) : 
            res.append(i), 
        i = i + 1
    return res

def get_closest_split(n, close_to=9000):
    all_divisors = getDivisors(n)
    for ix, val in enumerate(all_divisors):
        if close_to < val:
            if ix == 0: return val
            if (val-close_to)>(close_to - all_divisors[ix-1]):
                return all_divisors[ix-1]
            return val

def bin_spectrum(spectrum,rate):
    bin_size = math.ceil(5 * spectrum.size / rate)
    values_per_bin = math.ceil(spectrum.size / bin_size)
    padded_spectrum = np.pad(spectrum,(0,bin_size*values_per_bin-spectrum.size))
    binned_spectrum = np.mean(padded_spectrum.reshape(-1, bin_size), axis=1)

    x = np.arange(0,5*binned_spectrum.size//2 -2,5)
    plt.plot( x,np.abs(binned_spectrum[:(binned_spectrum.size//2)]))
    plt.show()
    sound_back = np.abs(ifft(binned_spectrum))
    wav.write('results/output_'+"out"+'.wav', rate, np.int16(sound_back* 32767 / sound_back.max()))


#Audio_file = 'Adele - Set Fire to the Rain'
plots = False
channel = 1             # for stereo songs we have two channels, choose one.

# Messing around - setting stuff to 0
cutoff_left = 0         # Hz - setting frequencies between cutoff_left and cutoff_right to 0
cutoff_right = 0        # Hz
threshold_cut = 0       # Setting all frequencies with a lower value than threshold_cut times max to 0
padding_offset = 0      # pad the signal with 0, front and back

def getSpectum(path,Song_name):

    rate, data = wav.read(path) # rate = sampling rate (typically 48000)
    print('Song name: ',Song_name)
    print('data shape (rows x channels) :',data.shape)
    if(len(data.shape) > 1):

        print('rate :',rate)

        # # pad signal
        # data = np.insert(data,0,np.repeat(np.array([[0,0]]),padding_offset,axis=0),axis=0)
        # data = np.append(data,np.repeat(np.array([[0,0]]),padding_offset,axis=0),axis=0)

        k = np.arange(data.shape[0])
        T = data.shape[0]/rate  
        frqLabel = k/T                  # Set correct x labels

        fft_out = fft(data[:,channel])
        
        # # setting stuff to 0
        # fft_out[np.where(frqLabel >= cutoff_left)[0][0]:np.where(frqLabel >= cutoff_right)[0][0]] = 0
        # fft_out[np.abs(fft_out)<np.abs(fft_out).max()*threshold_cut] = 0

        if (plots):
            show_duration = data.shape[0]
            # plt.title("Waveform of "+Song_name)
            # plt.plot(np.arange(show_duration),data[0:show_duration,channel])
            # plt.show()

            plt.title("Spectrum of "+Song_name)
            plt.plot(frqLabel[:(fft_out.size//2)], np.abs(fft_out[:(fft_out.size//2)]))
            # #plt.savefig('resuts/Spectrum_'+Song_name)
            plt.show()
        
        # cut out redunded information from fourier transform, namely the second half
        fft_out_cut = fft_out[:fft_out.size//2+1]
        
        return [fft_out_cut, rate]
    else:

        print('rate :',rate)

        # # pad signal
        # data = np.insert(data,0,np.repeat(np.array([0]),padding_offset,axis=0),axis=0)
        # data = np.append(data,np.repeat(np.array([0]),padding_offset,axis=0),axis=0)

        k = np.arange(data.shape[0])
        T = data.shape[0]/rate  
        frqLabel = k/T                  # Set correct x labels

        fft_out = fft(data)
        
        # # setting stuff to 0
        # fft_out[np.where(frqLabel >= cutoff_left)[0][0]:np.where(frqLabel >= cutoff_right)[0][0]] = 0
        # fft_out[np.abs(fft_out)<np.abs(fft_out).max()*threshold_cut] = 0
        
        if (plots):
            show_duration = data.shape[0]
            # plt.title("Waveform of "+Song_name)
            # plt.plot(np.arange(show_duration),data[0:show_duration])
            # plt.show()

            # plt.title("Spectrum of "+Song_name)
            # plt.plot(frqLabel[:(fft_out.size//2)], np.abs(fft_out[:(fft_out.size//2)]))
            # # plt.savefig('Spectrum_'+Song_name)
            # plt.show()
        
        # cut out redunded information from fourier transform, namely the second half
        fft_out_cut = fft_out[:fft_out.size//2+1]
        
        return [fft_out_cut, rate]


## if you want to create a sound file from the cut down version of the spectrum (fft_out_cut), you need to add the redundand second half first
# fft_out_cut_rejoined = np.concatenate((fft_out_cut,np.conjugate(fft_out_cut[1:-1])[::-1]))
# rate = 2000 # ? may need to adjust 
# sound_back = np.real(ifft(fft_out_cut_rejoined))
# wav.write('results/output_'+'rejoined'+'.wav', rate, np.int16(sound_back* 32767 / sound_back.max()))


# # try out fft and save back as an wav file -- for uncut fft
# fft_out,rate =  getSpectum("file_path", "some Song")
# sound_back = np.real(ifft(fft_out))
# wav.write('results/output_'+'working'+'.wav', rate, np.int16(sound_back* 32767 / sound_back.max()))
# if (plots):
#     show_duration = sound_back.shape[0]
#     plt.title("Back formation of audio file")
#     plt.plot(np.arange(show_duration),sound_back[0:show_duration])
#     plt.show()



## create binned spectrum -- not working  -- for uncut fft
# spectrum, rate = getSpectum("https://open.spotify.com/track/4aWmUDTfIPGksMNLV2rQP2", "data/temp", "some Song")
# bin_spectrum(spectrum,rate)
