from scipy.fftpack import fft, ifft
from scipy.io import wavfile as wav
import numpy as np
import matplotlib.pyplot as plt


# To create sample audio files and see effects of a fourier transform

# Settings:
plot = True    # set to True to only create plots with first frequency array. set to False to create soundfiles with second frequency array.
t_n = 5         # sec - track length
rate = 44100    # sample rate
N = t_n*rate    # datapoints
output_foulder = 'figures'

# Two different frequency arrays are needed, since Laptop speakers can only create hearable sound at around 80 Hz, 150 is better though
# These "high" frequencies aren't nice to plot though, therefore a different set of frequencies is used for the plot
# Also the sound length is set for 1 second
if(plot):
    t_n = 1
    N = t_n*rate
    frequencies = [4, 30, 60, 90]
else:
    frequencies = [150, 200, 250, 300]
    
 
def get_fft_values(y_values, rate, N):
    f_values = np.linspace(0.0, rate/(2.0), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values, fft_values_

xa = np.linspace(0, t_n, num=N)
xb = np.linspace(0, t_n/4., num=N//4)

y1a, y1b = np.sin(2*np.pi*frequencies[0]*xa), np.sin(2*np.pi*frequencies[0]*xb)
y2a, y2b = np.sin(2*np.pi*frequencies[1]*xa), np.sin(2*np.pi*frequencies[1]*xb)
y3a, y3b = np.sin(2*np.pi*frequencies[2]*xa), np.sin(2*np.pi*frequencies[2]*xb)
y4a, y4b = np.sin(2*np.pi*frequencies[3]*xa), np.sin(2*np.pi*frequencies[3]*xb)


composite_signal1 = y1a + y2a + y3a + y4a
composite_signal2 = np.concatenate([y1b, y2b, y3b, y4b])

f_values1, fft_values1, complex_fft_values1 = get_fft_values(composite_signal1, rate, N)
f_values2, fft_values2, complex_fft_values2 = get_fft_values(composite_signal2, rate, N)

sound_back_1 = np.real(ifft(complex_fft_values1))
sound_back_2 = np.real(ifft(complex_fft_values2))

if(plot):
    fig, axarr = plt.subplots(nrows=2, ncols=3, figsize=(16,9))
    cols = ['sound wave','frequency spectrum', 'reconstructed soundwave']
    rows = ['superposition of frequencies','concatenated frequencies']
    for ax, col in zip(axarr[0], cols):
        ax.set_title(col,  size='large')
    for ax, row in zip(axarr[:,0], rows):
        ax.set_ylabel(row ,size='large')    

    axarr[0,0].plot(xa, composite_signal1)
    axarr[0,1].plot(f_values1[0:160], fft_values1[0:160])
    axarr[0,2].plot(xa, sound_back_1)
    axarr[1,0].plot(xa, composite_signal2)
    axarr[1,1].plot(f_values2[0:160], fft_values2[0:160])
    axarr[1,2].plot(xa, sound_back_2)
    
    plt.tight_layout()
    plt.savefig(output_foulder + '/concatenated_vs_mixed_frequencies.pdf')
    print("saved figure")
else:
    wav.write(output_foulder + '/example_mixed.wav', rate , np.int16((composite_signal1 / composite_signal1.max())* 32767 ))
    wav.write(output_foulder + '/example_concatenated_.wav', rate , np.int16((composite_signal2 / composite_signal2.max())* 32767 ))
    wav.write(output_foulder + '/inverse_example_mixed.wav', rate , np.int16(sound_back_1* 32767 / sound_back_1.max()))
    wav.write(output_foulder + '/inverse_example_concatenated.wav', rate , np.int16(sound_back_2* 32767 / sound_back_2.max()))
    print("saved sound files")

