import numpy as np
from scipy.signal import spectrogram
import plots_aux as plt
from spect import spect



xlim = (0, 3)
sf = 1024


def ex1a_generate_signal(freq1=30, freq2=101, freq3=270, sf=sf, label='Generated Signal', show=False):
    ts1 = np.linspace(0, 1, int(sf), endpoint=False)
    signal1 = np.sin(2 * np.pi * freq1 * ts1)
    ts2 = np.linspace(1, 2, int(sf), endpoint=False)
    signal2 = np.sin(2 * np.pi * freq2 * ts2)
    ts3 = np.linspace(2, 3, int(sf), endpoint=False)
    signal3 = np.sin(2 * np.pi * freq3 * ts3)

    ts = np.concatenate((ts1, ts2, ts3))
    signal = np.concatenate((signal1, signal2, signal3))

    plt.plot_signal(ts, signal, label, show=show, xlim=xlim)
    return ts, signal


def ex1b_fft(signal=None, ts=None, label="FFT of generated signal", xticks=[30, 101, 270], show=False):
    if signal is None and ts is None:
        ts, signal = ex1a_generate_signal()

    pw, fq = spect(signal, sf)
    plt.plot_absolute_spectrum(fq, pw, label, xticks=xticks, show=show)


def ex1c_spectogram_psd(signal=None, ts=None, label="PSD of the generated signal", show=False):
    if signal is None and ts is None:
        ts, signal = ex1a_generate_signal()

    freqs, time, Sxx = spectrogram(signal, sf, window='hamming', nfft=len(signal), noverlap=0)
    plt.plot_spectogram(freqs, time, Sxx, label=label, show=show)


def ex1d_spectogram_stft(signal=None, ts=None, label="STFT of the generated signal", show=False):
    if signal is None and ts is None:
        ts, signal = ex1a_generate_signal()

    label1 = label + " -- Hamming window"
    freqs, time, Sxx = spectrogram(signal, sf, window='hamming', nfft=256, noverlap=0, mode='magnitude')
    plt.plot_spectogram(freqs, time, Sxx, label=label1, show=show)

    label2 = label + " -- Hann window"
    freqs, time, Sxx = spectrogram(signal, sf, window='hann', nfft=256, noverlap=0, mode='magnitude')
    plt.plot_spectogram(freqs, time, Sxx, label=label2, show=show)

    label3 = label + " -- Tukey window"
    freqs, time, Sxx = spectrogram(signal, sf, window=('tukey', 0.3), nfft=256, noverlap=0, mode='magnitude')
    plt.plot_spectogram(freqs, time, Sxx, label=label3, show=show)



#ex1a_generate_signal()
#ex1b_fft()
#ex1c_spectogram_psd()
ex1d_spectogram_stft()

