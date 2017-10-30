import pywt
from matplotlib import pyplot as plt


wavelet = pywt.ContinuousWavelet('gaus1')

for level in range(1):

    phi, psi = wavelet.wavefun(level=5)
    scaling_fun = phi
    wavelet_fun = psi

    plt.figure(0)
    plt.plot(phi,  label="lev"+str(level))
    plt.title('scaling function')
    plt.legend()

    plt.figure(1)
    plt.plot(psi,  label="lev"+str(level))
    plt.title('wavelet function')
    plt.legend()


plt.show()