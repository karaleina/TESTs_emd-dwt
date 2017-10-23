from PyEMD import EMD, EEMD
import numpy as np
import h5py
from matplotlib import pyplot as plt
from scipy.fftpack import fft
from scipy.signal import argrelextrema


filename_read = "data/breathing/camera/movements/K8.h5"


def count_fft(signal, T=0.3):
    new_y = signal.ravel()
    # FFT
    N = len(new_y)
    yf = fft(new_y)
    xf = np.linspace(0.0, 1.0 / (2 * T), N // 2)

    yf = yf / len(yf)
    yf_abs = np.abs(yf[0:len(yf) // 2])

    return [xf, yf_abs]

def calculate_max_peak(cropped_xf, cropped_filtered_fft, T_sampling=0.03):
    maxima = argrelextrema(cropped_filtered_fft, np.greater)
    maxima = maxima[0]
    true_maximas_freq = []
    if len(maxima) > 0:
        max_fft = None
        index_of_max_fft = -1
        for index in maxima:
            current_max_fft = cropped_filtered_fft[index]
            if max_fft == None or current_max_fft >= max_fft:
                max_fft = current_max_fft
                index_of_max_fft = index

        true_maximas_freq.append(index_of_max_fft)

        breath_freq = cropped_xf[index_of_max_fft]
        breath_ratio = breath_freq * 60

        return round(breath_ratio, 1)
    else:
        return None


with h5py.File(filename_read, 'r') as hf:
    all = hf['all'][:]
#
# print(all)
s = np.array(all.ravel())
s = s[:] # 0 - 400; 300 - 800; 600:

plt.figure(20000)
plt.plot(all)

emd = EMD()
IMFs = emd.emd(s)
print(IMFs)

plt.figure(20000).clear()
plt.plot(s)

plt.figure(0)
for i in range(len(IMFs)):

    plt.figure(0)
    plt.subplot(len(IMFs), 1, i + 1)
    plt.plot(IMFs[i])

    plt.figure(1)
    plt.subplot(len(IMFs), 1, i + 1)
    [x, f] = count_fft(IMFs[i].ravel(), T=0.03)
    breath_ratio = calculate_max_peak(x, f, T_sampling=0.03)
    plt.plot(x,f, label=str(breath_ratio) + "/min")
    plt.legend()


sygnal_trendu = IMFs[0] - s

# plt.figure(40000)
# plt.plot(sygnal_trendu)

plt.show()
