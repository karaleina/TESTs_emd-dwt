import h5py
import numpy as np
import pywt
from matplotlib import pyplot as plt
from scipy.fftpack import fft
from scipy.signal import argrelextrema
from filters import filters
from scipy.signal import freqz


def count_fft(signal, T=0.01):
        new_y = signal.ravel()
        # FFT
        N = len(new_y)

        yf = fft(new_y)
        xf = np.linspace(0.0, 1.0 / (2 * T), N // 2)

        yf = yf / len(yf)
        yf_abs = np.abs(yf[0:len(yf) // 2])

        return [xf, yf_abs]


def simple_plotting(breathing_signal, fs, lowcut=0.08, highcut = 2.0):
    if breathing_signal == None:
        print("Brak oddechu")

        plt.figure(1).clear()
        plt.grid()
        plt.title("Signal is none :( ")

    else:
        breathing_signal = breathing_signal.ravel()

        plt.figure(1).clear()
        plt.subplot(2, 1, 1)

        x = np.linspace(0, len(breathing_signal)-1,num=len(breathing_signal))
        plt.plot(x, breathing_signal)
        plt.grid()
        plt.title("Breath signal")
        plt.xlabel("[n]")
        plt.tight_layout()

        additional_zeros = np.zeros(0*len(breathing_signal))
        breathing_signal = np.concatenate([breathing_signal, additional_zeros])

        xf, yf_abs = count_fft(breathing_signal)

        # Filtration
        for order in [4]:
            b, a = filters.butter_bandpass(lowcut, highcut, fs, order=order)
            w, h = freqz(b, a, worN=len(yf_abs))
            abs_filter = abs(h)

        filtered_fft = abs_filter * yf_abs

        # Cropped filtration

        cropped_xf = xf[xf <= 0.8 * highcut]
        cropped_filtered_fft = filtered_fft[0:len(cropped_xf)]
        cropped_xf = cropped_xf[0:len(cropped_xf // 2)]

        # Peaks
        plt.subplot(2, 1, 2).cla()
        plt.plot(cropped_xf, cropped_filtered_fft, "*-")
        plt.xlabel("Frequency [Hz]")

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
            breath_signal = breath_freq * 60
            plt.title("FFT (breath rate: " + str(round(float(breath_signal), 2)) + "1/s)")

            max_fft_signal = cropped_filtered_fft[index_of_max_fft]
            for max_index in maxima:
                plt.plot(cropped_xf[max_index], cropped_filtered_fft[max_index], "bo")

                if cropped_filtered_fft[max_index] > 1 / 3 * max_fft_signal and max_index != index_of_max_fft:
                    true_maximas_freq.append(cropped_xf[max_index])
        else:
            print("Nie znaleziono maximów!")
        plt.grid()

        plt.tight_layout()
        return true_maximas_freq
#
# def filter_fft(xf, yf, Ts=0.03, lowcut=0.02, highcut=2):
#     for order in [4]:
#         b, a = filters.butter_bandpass(lowcut, highcut, fs=1/Ts, order=order)
#         w, h = freqz(b, a, worN=len(yf))
#         #plt.plot((1/Ts * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
#         abs_filter = abs(h)
#
#     filtered_fft = abs_filter * yf
#     cropped_xf = xf[xf <= highcut]
#     cropped_filtered_fft = filtered_fft[0:len(cropped_xf)]
#     cropped_xf = cropped_xf[0:len(cropped_xf // 2)]
#
#     return (cropped_xf, cropped_filtered_fft)
#
#
# def calculate_max_peak(cropped_xf, cropped_filtered_fft):
#     maxima = argrelextrema(cropped_filtered_fft, np.greater)
#     maxima = maxima[0]
#     true_maximas_freq = []
#     if len(maxima) > 0:
#         max_fft = None
#         index_of_max_fft = -1
#         for index in maxima:
#             current_max_fft = cropped_filtered_fft[index]
#             if max_fft == None or current_max_fft >= max_fft:
#                 max_fft = current_max_fft
#                 index_of_max_fft = index
#
#         true_maximas_freq.append(index_of_max_fft)
#
#         breath_freq = cropped_xf[index_of_max_fft]
#         breath_ratio = breath_freq * 60
#
#         return round(breath_ratio, 3)
#     else:
#         return None
#
#

def discard_the_constant(signal, wavelet="db6"):

    try:
        (cA, cD) = pywt.dwt(signal, wavelet)
        (cA2, cD2) = pywt.dwt(cA, wavelet)
        (cA3, cD3) = pywt.dwt(cA2, wavelet)
        (cA4, cD4) = pywt.dwt(cA3, wavelet)
        (cA5, cD5) = pywt.dwt(cA4, wavelet)
        (cA6, cD6) = pywt.dwt(cA5, wavelet)
        (cA7, cD7) = pywt.dwt(cA6, wavelet)
        (cA8, cD8) = pywt.dwt(cA7, wavelet)
        (cA9, cD9) = pywt.dwt(cA8, wavelet)
        (cA10, cD10) = pywt.dwt(cA9, wavelet)

        cA10 = np.zeros(len(cA10))

        list_of_coeffs = [cA10, cD10, cD9, cD8, cD7, cD6,
                          cD5, cD4, cD3, cD2, cD]

        return pywt.waverec(list_of_coeffs, wavelet)

    except ValueError:
        print("Wavelet transform except even number of samples only.")

def filter_100Hz_signal(signal, wavelet="db6"):

    try:
        (cA, cD) = pywt.dwt(signal, wavelet)
        (cA2, cD2) = pywt.dwt(cA, wavelet)
        (cA3, cD3) = pywt.dwt(cA2, wavelet)
        (cA4, cD4) = pywt.dwt(cA3, wavelet)
        (cA5, cD5) = pywt.dwt(cA4, wavelet)

        cD = np.zeros(len(cD))
        cD2 = np.zeros(len(cD2))
        cD3 = np.zeros(len(cD3))
        cD4 = np.zeros(len(cD4))
        cD5 = np.zeros(len(cD5))

        list_of_coeffs = [cA5, cD5, cD4, cD3, cD2, cD]

        return pywt.waverec(list_of_coeffs, wavelet)

    except ValueError:
        print("Wavelet transform except even number of samples only.")

filename_read = "data/breathing/camera/movements_ref/oddech_w_ruchu_ref_K_7.h5"
with h5py.File(filename_read, 'r') as hf:
    all = hf['all'][:]

s = np.array(all.ravel())
print(len(s))
start = 200
s = s[start:len(s)-500]
s = discard_the_constant(s, wavelet="db6")
breathing_signal = filter_100Hz_signal(s, wavelet="db6")
# additional_zeros = np.zeros(0 * len(breathing_signal))
# breathing_signal = np.concatenate([breathing_signal, additional_zeros])

simple_plotting(breathing_signal=breathing_signal, fs=100, lowcut=0.02, highcut=2.0)
plt.show()



