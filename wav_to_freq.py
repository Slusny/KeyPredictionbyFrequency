import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft, ifft
import numpy as np
import math

Audio_file = 'Adele - Set Fire to the Rain'
plots = True
channel = 1             # for stereo songs we have two channels, choose one.

# Messing around - setting stuff to 0
cutoff_left = 0         # Hz - setting frequencies between cutoff_left and cutoff_right to 0
cutoff_right = 0        # Hz
threshold_cut = 0       # Setting all frequencies with a lower value than threshold_cut times max to 0
padding_offset = 0      # pad the signal with 0, front and back

def getSpectum(Audio_file):

    rate, data = wav.read(Audio_file + '.wav') # rate = sampling rate (typically 48000)

    print('data shape (rows x channels) :',data.shape)
    if(len(data.shape) > 1):

        print('rate :',rate)

        # pad signal
        data = np.insert(data,0,np.repeat(np.array([[0,0]]),padding_offset,axis=0),axis=0)
        data = np.append(data,np.repeat(np.array([[0,0]]),padding_offset,axis=0),axis=0)

        k = np.arange(data.shape[0])
        T = data.shape[0]/rate  
        frqLabel = k/T                  # Set correct x labels

        fft_out = fft(data[:,channel])
        # setting stuff to 0
        fft_out[np.where(frqLabel >= cutoff_left)[0][0]:np.where(frqLabel >= cutoff_right)[0][0]] = 0
        fft_out[np.abs(fft_out)<np.abs(fft_out).max()*threshold_cut] = 0

        if (plots):
            show_duration = data.shape[0]
            plt.title("Waveform of "+Audio_file)
            plt.plot(np.arange(show_duration),data[0:show_duration,channel])
            plt.show()

            plt.title("Spectrum of "+Audio_file)
            plt.plot(frqLabel[:(fft_out.size//2)], np.abs(fft_out[:(fft_out.size//2)]))
            plt.savefig('resuts/Spectrum_'+Audio_file)
            plt.show()
        
        return [fft_out, rate]
    else:

        print('rate :',rate)

        # pad signal
        data = np.insert(data,0,np.repeat(np.array([0]),padding_offset,axis=0),axis=0)
        data = np.append(data,np.repeat(np.array([0]),padding_offset,axis=0),axis=0)

        k = np.arange(data.shape[0])
        T = data.shape[0]/rate  
        frqLabel = k/T                  # Set correct x labels

        fft_out = fft(data)
        # setting stuff to 0
        fft_out[np.where(frqLabel >= cutoff_left)[0][0]:np.where(frqLabel >= cutoff_right)[0][0]] = 0
        fft_out[np.abs(fft_out)<np.abs(fft_out).max()*threshold_cut] = 0
        if (plots):
            show_duration = data.shape[0]
            plt.title("Waveform of "+Audio_file)
            plt.plot(np.arange(show_duration),data[0:show_duration])
            plt.show()

            plt.title("Spectrum of "+Audio_file)
            plt.plot(frqLabel[:(fft_out.size//2)], np.abs(fft_out[:(fft_out.size//2)]))
            plt.savefig('Spectrum_'+Audio_file)
            plt.show()
        
        return [fft_out, rate]


# save back as an wav file
fft_out,rate =  getSpectum(Audio_file)
sound_back = np.abs(ifft(fft_out))
wav.write('resuts/output_'+Audio_file+'.wav', rate, np.int16(sound_back* 32767 / sound_back.max()))
if (plots):
    show_duration = sound_back.shape[0]
    plt.title("Back formation of audio file")
    plt.plot(np.arange(show_duration),sound_back[0:show_duration])
    plt.show()


## Overlaying two Spectren, Adele was way more dominant and sounded awefull :D

# Adele,rate = getSpectum('Adele - Set Fire to the Rain')
# Weekend,rate = getSpectum('The Weeknd - Blinding Lights')
# if(Adele.size < Weekend.size):
#     Adele.resize(Weekend.shape)
# else:
#     Weekend.resize(Adele.shape)

# fft_out = Adele + Weekend

# sound_back = np.abs(ifft(fft_out))
# if (plots):
#     show_duration = sound_back.shape[0]
#     plt.plot(np.arange(show_duration),sound_back[0:show_duration])
#     plt.show()
# wav.write('output.wav', rate, np.int16(sound_back* 32767 / sound_back.max()))




## old attempt on binning the spectrum - not working but may be repurposed

# def getDivisors(n, res=None) : 
#     res = res or []
#     i = 1
#     while i <= n : 
#         if (n % i==0) : 
#             res.append(i), 
#         i = i + 1
#     return res

# def get_closest_split(n, close_to=9000):
#     all_divisors = getDivisors(n)
#     for ix, val in enumerate(all_divisors):
#         if close_to < val:
#             if ix == 0: return val
#             if (val-close_to)>(close_to - all_divisors[ix-1]):
#                 return all_divisors[ix-1]
#             return val


#     size = get_closest_split(data.shape[0])
#     bins = np.arange(size)

#     data_2 = data[:,channel].reshape(-1,size).mean(axis=1)
#     fft_out_2 = fft_out.reshape(-1,size).mean(axis=1)

#     x = np.arange(data_2.size)

#     #plt.hist( np.abs(fft_out_2),bins)
#     #plt.scatter(data_2, np.abs(fft_out_2),alpha=0.3)
#     #plt.scatter(x, data_2,alpha=0.3)