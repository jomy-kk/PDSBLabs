import numpy as np
from scipy.signal import spectrogram, get_window
import plots_aux as plt
from spect import spect



xlim = (0, 3)
sf = 1024


def ex1a_generate_signal(freq1=30, freq2=101, freq3=270, sf=sf, label='Signal A', show=False):
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


def ex1b_fft(signal=None, ts=None, label="FFT of signal A", xticks=[30, 101, 270], show=False):
    if signal is None and ts is None:
        ts, signal = ex1a_generate_signal()

    pw, fq = spect(signal, sf)
    plt.plot_absolute_spectrum(fq, pw, label, xticks=xticks, show=show)


def ex1c_spectrogram(signal=None, ts=None, label="Amplitude spectrogram of signal A", show=False):
    if signal is None and ts is None:
        ts, signal = ex1a_generate_signal(show=False)

    wlen = len(signal) - 1499
    freqs, time, Sxx = spectrogram(signal, sf, window=get_window('hamming', wlen), nperseg=wlen, nfft=wlen, noverlap=0, mode='psd')
    plt.plot_spectogram(freqs, time, Sxx, label="Hamming (" + str(wlen) + ")", show=show)


def ex1d_f_g_spectrogram_stft(signal=None, ts=None, wlen=256, overlaps=(0,), show=False):
    if signal is None and ts is None:
        ts, signal = ex1a_generate_signal()

    for overlap in overlaps:
        overlap = int(overlap * wlen)

        label1 = "Hamming window, size " + str(wlen) + ", overlap " + str(overlap)
        freqs, time, Sxx = spectrogram(signal, sf, window=get_window('hamming', wlen), nperseg=wlen, noverlap=overlap, mode='magnitude')
        plt.plot_spectogram(freqs, time, Sxx, label=label1, show=show)

        label2 = "Hann window, size " + str(wlen) + ", overlap " + str(overlap)
        freqs, time, Sxx = spectrogram(signal, sf, window=get_window('hann', wlen), nperseg=wlen, noverlap=overlap, mode='magnitude')
        plt.plot_spectogram(freqs, time, Sxx, label=label2, show=show)

        label3 = "Tukey window, size " + str(wlen) + ", overlap " + str(overlap)
        freqs, time, Sxx = spectrogram(signal, sf, window=('tukey', 0.25), nperseg=wlen, noverlap=overlap, mode='magnitude')
        plt.plot_spectogram(freqs, time, Sxx, label=label3, show=show)



#ex1a_generate_signal()
#ex1b_fft()
#ex1c_spectrogram(show=True)
#ex1d_spectrogram_stft()
#ex1d_f_g_spectrogram_stft(wlen=256)
#ex1d_f_g_spectrogram_stft(wlen=256, overlaps=(0.1, 0.5, 0.7))
#ex1d_f_g_spectrogram_stft(wlen=64, overlaps=(0.1, 0.5, 0.7))

