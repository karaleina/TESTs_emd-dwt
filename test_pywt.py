from matplotlib import pyplot as plt
import numpy as np
import pywt
import h5py
from scipy.fftpack import fft
from scipy.signal import argrelextrema


def count_fft(signal, T=0.3):
    new_y = signal.ravel()
    # new_y = np.zeros(3*len(old_y))
    # new_y[0:len(old_y)] = old_y

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


def filter_signal(signal, wavelet="db6"):

    try:
        (cA, cD) = pywt.dwt(signal, wavelet)
        (cA2, cD2) = pywt.dwt(cA, wavelet)
        (cA3, cD3) = pywt.dwt(cA2, wavelet)

        cD = np.zeros(len(cD))
        cD2 = np.zeros(len(cD2))
        cD3 = np.zeros(len(cD3))

        list_of_coeffs = [cA3, cD3, cD2, cD]

        return pywt.waverec(list_of_coeffs, wavelet)

    except ValueError:
        print("Wavelet transform except even number of samples only.")


def filter_90Hz_signal(signal, wavelet="db6"):

    try:
        (cA, cD) = pywt.dwt(signal, wavelet)
        (cA2, cD2) = pywt.dwt(cA, wavelet)
        (cA3, cD3) = pywt.dwt(cA2, wavelet)
        (cA4, cD4) = pywt.dwt(cA3, wavelet)

        cD = np.zeros(len(cD))
        cD2 = np.zeros(len(cD2))
        cD3 = np.zeros(len(cD3))
        cD4 = np.zeros(len(cD4))

        list_of_coeffs = [cA4, cD4, cD3, cD2, cD]

        return pywt.waverec(list_of_coeffs, wavelet)

    except ValueError:
        print("Wavelet transform except even number of samples only.")
filename_read = "data/breathing/camera/movements/K5.h5"


with h5py.File(filename_read, 'r') as hf:
    all = hf['all'][:]

s = np.array(all.ravel())
 # 0 - 400; 300 - 800; 600:
#
# #Dodam szum
# noise = np.random.random(len(s))
# noise = np.multiply(10, noise)
# noisy_signal = np.add(s, noise)
#
# plt.figure(-1)
# plt.plot(noisy_signal)

plt.figure(0)
plt.subplot(2,1,1)
plt.plot(s)
plt.subplot(2,1,2)
[x, f] = count_fft(s.ravel(), T=0.03)
breath_ratio = calculate_max_peak(x, f, T_sampling=0.03)
plt.plot(x, f, label=str(breath_ratio) + "/min")
plt.legend()

new_s = filter_signal(s, wavelet="db6")

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(new_s)
[x, f] = count_fft(new_s.ravel(), T=0.03)
breath_ratio = calculate_max_peak(x, f, T_sampling=0.03)
plt.subplot(2,1,2)
plt.plot(x, f, label=str(breath_ratio) + "/min")
plt.legend()

plt.show()

# TODO rsyowanie współczynników
