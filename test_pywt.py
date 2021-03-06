from matplotlib import pyplot as plt
import numpy as np
import pywt
import h5py
from scipy.fftpack import fft
from scipy.signal import argrelextrema
from filters import filters
from scipy.signal import freqz


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


def simple_plotting(breathing_signal, fs, lowcut=0.08, highcut = 2.0):
    if breathing_signal == None:
        print("Brak oddechu")
        plt.grid()
        plt.title("Signal is none :( ")

    else:
        breathing_signal = breathing_signal.ravel()


        plt.subplot(2, 1, 1)

        x = np.linspace(0, len(breathing_signal)-1,num=len(breathing_signal))
        plt.plot(x, breathing_signal)
        plt.grid()
        plt.title("Breath signal")
        plt.xlabel("[n]")
        plt.tight_layout()


        additional_zeros = np.zeros(3*len(breathing_signal))
        breathing_signal = np.concatenate([breathing_signal, additional_zeros])

        xf, yf_abs = count_fft(breathing_signal, T=1/fs)

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
            plt.title("FFT (breath rate: " + str(round(float(breath_signal), 2)) + "/min)")

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


# #########################################################

filename_read = "data/breathing/camera/movements/K7.h5"
with h5py.File(filename_read, 'r') as hf:
    all = hf['all'][:]

s = np.array(all.ravel())
plt.figure(0)
simple_plotting(s, fs = 1/0.03)

new_s = filter_signal(s, wavelet="haar")
plt.figure(1)
simple_plotting(new_s, fs=1/0.03)

# plt.figure(0)
# plt.subplot(2,1,1)
# plt.plot(s)
# plt.subplot(2,1,2)
# [x, f] = count_fft(s.ravel(), T=0.03)
# breath_ratio = calculate_max_peak(x, f, T_sampling=0.03)
# plt.plot(x, f, label=str(breath_ratio) + "/min")
# plt.legend()
# plt.figure(1)
# plt.subplot(2,1,1)
# plt.plot(new_s)
# [x, f] = count_fft(new_s.ravel(), T=0.03)
# breath_ratio = calculate_max_peak(x, f, T_sampling=0.03)
# plt.subplot(2,1,2)
# plt.plot(x, f, label=str(breath_ratio) + "/min")
# plt.legend()
#
plt.show()

# TODO rsyowanie współczynników
