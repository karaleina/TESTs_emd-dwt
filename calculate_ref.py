import h5py
import numpy as np
import pywt
from matplotlib import pyplot as plt
from scipy.fftpack import fft
from scipy.signal import argrelextrema
from filters import filters
from scipy.signal import freqz


def count_fft(signal, T=0.3):
    new_y = signal.ravel()
    # FFT
    N = len(new_y)
    yf = fft(new_y)
    xf = np.linspace(0.0, 1.0 / (2 * T), N // 2)

    yf = yf / len(yf)
    yf_abs = np.abs(yf[0:len(yf) // 2])

    return [xf, yf_abs]


def filter_fft(xf, yf, Ts=0.03, lowcut=0.02, highcut=2):
    for order in [4]:
        b, a = filters.butter_bandpass(lowcut, highcut, fs=1/Ts, order=order)
        w, h = freqz(b, a, worN=len(yf))
        #plt.plot((1/Ts * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        abs_filter = abs(h)

    filtered_fft = abs_filter * yf
    cropped_xf = xf[xf <= 0.8 * highcut]
    cropped_filtered_fft = filtered_fft[0:len(cropped_xf)]
    cropped_xf = cropped_xf[0:len(cropped_xf // 2)]

    return (cropped_xf, cropped_filtered_fft)


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

        return round(breath_ratio, )
    else:
        return None


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


filename_read = "data/breathing/camera/movements_ref/oddech_w_ruchu_ref_K_5.h5"
with h5py.File(filename_read, 'r') as hf:
    all = hf['all'][:]

s = np.array(all.ravel())
print(len(s))
start = 200
s = s[start:len(s)-500]

# PLOTTING
plt.figure(0)
plt.subplot(3,1,1)
plt.plot(s)
plt.grid()

plt.figure(0)
plt.subplot(3,1,2)
filtred_s = filter_100Hz_signal(s, wavelet="db6")
plt.plot(filtred_s)
plt.grid()

plt.figure(0)
plt.subplot(3,1,3)
xf, y_fft = count_fft(filtred_s, T = 0.01)
xf, y_fft = filter_fft(xf, y_fft, Ts=0.03, lowcut=0.02, highcut=2)
breath_rate = calculate_max_peak(xf, y_fft, T_sampling=0.01)
plt.plot(xf,y_fft)
plt.title(str(breath_rate) + "/min")
plt.grid()

plt.show()



