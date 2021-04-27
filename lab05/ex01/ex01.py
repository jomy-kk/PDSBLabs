import numpy as np

import aux
from pywt import wavedec, waverecn, cwt
from matplotlib import pyplot as plt


def ex1_wavelet_decomposition(t, signal, n_stages, wavelet, xlim, label, show=False):
    coeffs = wavedec(signal, wavelet, level=n_stages)
    reconstructed_signal = waverecn(coeffs, wavelet)

    for l in xlim:
        aux.plot_signal_processed(t, signal, t, reconstructed_signal, 'Reconstructed', label, xlim=l, show=show)




(sA, sf_s16), (sB, sf_s16), (sC, sf_s16), (ecg_signal, sf_ecg) = aux.get_signals()
t = np.linspace(1, len(sA), len(sA))


ex1_wavelet_decomposition(t, sA, 6, 'db1', xlim=((0, 4096), (1000, 1200), (2000, 2200), (3000, 3200)),
                          label='Ex 1 - Signal A', show=False)
ex1_wavelet_decomposition(t, sB, 6, 'db1', xlim=((0, 4096), (1000, 1200), (2000, 2200), (3000, 3200)),
                          label='Ex 1 - Signal B', show=False)
ex1_wavelet_decomposition(t, sC, 6, 'db1', xlim=((0, 4096), (1000, 1200), (2000, 2200), (3000, 3200)),
                          label='Ex 1 - Signal C', show=False)

def ex2_scalogram(t, signal, sf, wavelet, n_stages):

    ig, ax = plt.subplots(figsize=(12, 8))
    coeffs = wavedec(signal, wavelet, level=n_stages)
    reconstructed_signal = waverecn(coeffs, wavelet)

    ax.plot(t, signal, color="b", alpha=0.5, label='original signal')
    # rec = lowpassfilter(signal, 0.4)
    ax.plot(t, reconstructed_signal, 'k', label='DWT smoothing}', linewidth=2)
    ax.legend()
    ax.set_title('Removing High Frequency Noise with DWT', fontsize=18)
    ax.set_ylabel('Signal Amplitude', fontsize=16)
    ax.set_xlabel('Sample No', fontsize=16)
    plt.margins(0)
    plt.show()

    # Scalogram
    scale_range = np.arange(2, 50)
    dt = 1/sf
    coefficients, frequencies = cwt(signal, scale_range, 'db1', dt)

    power = (abs(coefficients)) ** 2
    period = frequencies
    levels = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
    contourlevels = np.log2(levels)  # original
    time = range(4096)

    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both', cmap=plt.cm.seismic)

    ax.set_title("CWT of Signal", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=18)
    ax.set_xlabel("Samples", fontsize=18)
    yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))  # original
    ax.set_yticklabels(yticks)  # original
    ax.invert_yaxis()
    ylim = ax.get_ylim()

    # cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    # fig.colorbar(im, cax=cbar_ax, orientation="vertical")

    return yticks, ylim







