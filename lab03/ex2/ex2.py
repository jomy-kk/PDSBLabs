import numpy as np
from scipy.io.wavfile import write
from scipy.signal import resample, decimate
import plots_aux as plt

xlim = (0, 0.01)

def ex2_1_generate_sinusoid(frequency=2000, duration=3, sampling_frequency=22400, label='Signal X', save=False, plot=False, show=False):
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
    plt.plot_signal_processed(ts, signal, resampled_ts, resampled_signal, 'Downsampled', 'Downsampled with factor ' + str(factor), show=show, xlim=xlim)

def ex2_3_decimate():
    pass


def ex2_4_absolute_spectrum():
    pass


def ex2_6_interpolate():
    pass


def ex2_8_resample():
    pass


#ex2_1_generate_sinusoid(save=True, plot=True, show=False)
ex2_2_downsample(factor=2, show=False)
ex2_2_downsample(factor=4, show=False)
ex2_2_downsample(factor=8, show=False)
ex2_2_downsample(factor=16, show=False)
