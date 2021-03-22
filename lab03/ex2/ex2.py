import numpy as np
from scipy.io.wavfile import write, read
from scipy.signal import resample, decimate, upfirdn, firwin, kaiserord, cheby2, cheb2ord
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
    sampling_frequency = sf/factor
    resampled_ts = np.linspace(0, len(resampled_signal)/sampling_frequency, len(resampled_signal), endpoint=False)
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


def ex2_6_interpolate(signal=None, ts=None, up_factor=2, down_factor=11, sampling_frequency=sf, ftype='fir', show=False):
    if signal is None and ts is None:
        ts, signal = ex2_1_generate_sinusoid()
        sampling_frequency = sf

    if signal is not None and ts is None:
        ts = np.linspace(0, len(signal)/sf, len(signal), endpoint=False)

    # just do the beginning (bc of time complexity)
    ts, signal = ts[:5000], signal[:5000]

    if ftype == 'fir':
        nyq_rate = sampling_frequency / 2
        width = 400.0 / nyq_rate  # with a 400 Hz transition width
        ripple_db = 60.0  # attenuation in the stop band (dB)
        filter_order, beta = kaiserord(ripple_db, width)  # order and Kaiser parameter for the FIR filter

        cutoff = 0.5/up_factor
        filter_taps = firwin(filter_order, cutoff*nyq_rate, window=('kaiser', beta), fs=sampling_frequency)

        print("Filter order", filter_order)
        resampled_signal = upfirdn(filter_taps, signal, up=up_factor, down=down_factor)

    if ftype == 'cheby2':
        pass
        # TODO: with Chebychev II filter


    sampling_frequency *= up_factor/down_factor
    resampled_ts = np.linspace(0, len(resampled_signal)/sampling_frequency, len(resampled_signal), endpoint=False)
    plt.plot_signal_processed(ts, signal, resampled_ts, resampled_signal,
                              'Interpolated', 'Interpolated with factor ' + str(up_factor) + '/' + str(down_factor),
                              show=show, xlim=xlim)
    return resampled_ts, resampled_signal


def ex2_8_resample(new_sampling_frequency, show=False):
    sampling_frequency, signal = read("../data/tone.wav")
    n_samples = len(signal)
    duration = n_samples/sampling_frequency  # in seconds
    new_n_samples = int(n_samples * new_sampling_frequency / sampling_frequency)
    resampled_signal = resample(signal, new_n_samples)
    ts = np.linspace(0, duration, n_samples, endpoint=False)
    resampled_ts = np.linspace(0, duration, new_n_samples, endpoint=False)
    plt.plot_signal_processed(ts, signal, resampled_ts, resampled_signal,
                              'Resampled', 'Resampled to ' + str(new_sampling_frequency) + ' Hz', xlim=xlim, show=show)
    return resampled_signal


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


ex2_4_absolute_spectrum(ex2_1_generate_sinusoid()[1], sf, "Original", show=False)
ex2_4_absolute_spectrum(ex2_2_downsample(factor=2)[1], sf/2, "Downsampled by 2", show=False)
ex2_4_absolute_spectrum(ex2_2_downsample(factor=4)[1], sf/4, "Downsampled by 4", show=False)
ex2_4_absolute_spectrum(ex2_2_downsample(factor=8)[1], sf/8, "Downsampled by 8", show=False)
ex2_4_absolute_spectrum(ex2_2_downsample(factor=16)[1], sf/16, "Downsampled by 16", show=False)
ex2_4_absolute_spectrum(ex2_3_decimate(factor=2)[1], sf/2, "Decimated by 2", show=False)
ex2_4_absolute_spectrum(ex2_3_decimate(factor=4)[1], sf/4, "Decimated by 4", show=False)
ex2_4_absolute_spectrum(ex2_3_decimate(factor=8)[1], sf/8, "Decimated by 8", show=False)
ex2_4_absolute_spectrum(ex2_3_decimate(factor=16)[1], sf/16, "Decimated by 16", show=False)
'''

#ex2_6_interpolate(up_factor=2, down_factor=11, ftype='fir', show=False)
#ex2_6_interpolate(up_factor=4, down_factor=23, ftype='fir', show=False)
#sf, signal = read('../data/looneyTunes.wav')
#xlim = (0.5, 0.52)
#ex2_6_interpolate(signal=signal, sampling_frequency=sf, up_factor=3, down_factor=8, ftype='fir', show=False)

#ex2_8_resample(7700, show=True)
#ex2_8_resample(4100, show=True)

