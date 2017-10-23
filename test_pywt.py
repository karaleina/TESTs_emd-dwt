from matplotlib import pyplot as plt
import numpy as np
import pywt
import h5py

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
filename_read = "data/camera/movements/K1.h5"

with h5py.File(filename_read, 'r') as hf:
    all = hf['all'][:]

s = np.array(all.ravel())
 # 0 - 400; 300 - 800; 600:
#
#Dodam szum
noise = np.random.random(len(s))
noise = np.multiply(10, noise)
noisy_signal = np.add(s, noise)

plt.figure(-1)
plt.plot(noisy_signal)

new_s = filter_90Hz_signal(noisy_signal, wavelet="db6")
plt.figure(1)
plt.plot(new_s)
plt.show()

# TODO rsyowanie współczynników
