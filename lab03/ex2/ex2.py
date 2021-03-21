import numpy as np
from scipy.io.wavfile import write
from scipy.signal import resample, decimate, spectrogram
from scipy.fft import fft
import plots_aux as plt
import math

xlim = (0, 0.01)
sf = 22400

def ex2_1_generate_sinusoid(frequency=2000, duration=3, sampling_frequency=sf, label='Signal X', save=False, plot=False, show=False):
    samples = np.linspace(0, duration, int(sampling_frequency * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * samples)
    if save:
        write('2k.wav', sampling_frequency, signal)
    if plot:
        plt.plot_signal(samples, signal, label, show=show, xlim=xlim)
    return samples, signal


def ex2_2_downsample(signal=None, ts=None, factor=2, show=False):
    if signal is None and ts is None:
        ts, signal = ex2_1_generate_sinusoid()
    resampled_signal, resampled_ts = resample(signal, int(len(signal)/factor), ts)
    plt.plot_signal_processed(ts, signal, resampled_ts, resampled_signal,
                              'Downsampled', 'Downsampled with factor ' + str(factor), show=show, xlim=xlim)
    return resampled_ts, resampled_signal


def ex2_3_decimate(signal=None, ts=None, factor=2, show=False):
    if signal is None and ts is None:
        ts, signal = ex2_1_generate_sinusoid()
    resampled_signal = decimate(signal, factor, zero_phase=True)
    resampled_ts = np.linspace(0, len(resampled_signal)/sf, len(resampled_signal), endpoint=False)
    plt.plot_signal_processed(ts, signal, resampled_ts, resampled_signal, 'Decimated',
                              'Decimated with factor ' + str(factor), show=show, xlim=xlim)
    return resampled_ts, resampled_signal


def spect(signal, sf):
    l = len(signal)
    p = abs(fft(signal) / l)
    pw = p[1:math.floor(l / 2) + 1]
    pw[2: - 1] = 2 * pw[2: - 1]
    fq = sf * np.arange(l/2-1) / l
    if len(pw) > len(fq):
        pw = pw[:len(fq)]
    else:
        fq = fq[:len(pw)]
    return pw, fq


def ex2_4_absolute_spectrum(signal, sampling_frequency, label, show=False):
    assert signal is not None
    pw, fq, = spect(signal, sampling_frequency)
    plt.plot_absolute_spectrum(fq, pw, label, show=show)


def ex2_6_interpolate():
    pass


def ex2_8_resample():
    pass


#ex2_1_generate_sinusoid(save=True, plot=True, show=False)

#ex2_2_downsample(factor=2, show=False)
#ex2_2_downsample(factor=4, show=False)
#ex2_2_downsample(factor=8, show=False)
#ex2_2_downsample(factor=16, show=False)

#ex2_3_decimate(factor=2)
#ex2_3_decimate(factor=4)
#ex2_3_decimate(factor=8)
#ex2_3_decimate(factor=16)

'''
ex2_4_absolute_spectrum(ex2_1_generate_sinusoid()[1], sf, "Original", show=False)
ex2_4_absolute_spectrum(ex2_2_downsample(factor=2)[1], sf, "Downsampled by 2", show=False)
ex2_4_absolute_spectrum(ex2_2_downsample(factor=4)[1], sf, "Downsampled by 4", show=False)
ex2_4_absolute_spectrum(ex2_2_downsample(factor=8)[1], sf, "Downsampled by 8", show=False)
ex2_4_absolute_spectrum(ex2_2_downsample(factor=16)[1], sf, "Downsampled by 16", show=False)
ex2_4_absolute_spectrum(ex2_3_decimate(factor=2)[1], sf, "Decimated by 2", show=False)
ex2_4_absolute_spectrum(ex2_3_decimate(factor=4)[1], sf, "Decimated by 4", show=False)
ex2_4_absolute_spectrum(ex2_3_decimate(factor=8)[1], sf, "Decimated by 8", show=False)
ex2_4_absolute_spectrum(ex2_3_decimate(factor=16)[1], sf, "Decimated by 16", show=False)
'''

ex2_4_absolute_spectrum(ex2_1_generate_sinusoid()[1], sf, "Original", show=False)
ex2_4_absolute_spectrum(ex2_2_downsample(factor=2)[1], sf/2, "Downsampled by 2", show=False)
ex2_4_absolute_spectrum(ex2_2_downsample(factor=4)[1], sf/4, "Downsampled by 4", show=False)
ex2_4_absolute_spectrum(ex2_2_downsample(factor=8)[1], sf/8, "Downsampled by 8", show=False)
ex2_4_absolute_spectrum(ex2_2_downsample(factor=16)[1], sf/16, "Downsampled by 16", show=False)
ex2_4_absolute_spectrum(ex2_3_decimate(factor=2)[1], sf/2, "Decimated by 2", show=False)
ex2_4_absolute_spectrum(ex2_3_decimate(factor=4)[1], sf/4, "Decimated by 4", show=False)
ex2_4_absolute_spectrum(ex2_3_decimate(factor=8)[1], sf/8, "Decimated by 8", show=False)
ex2_4_absolute_spectrum(ex2_3_decimate(factor=16)[1], sf/16, "Decimated by 16", show=False)
