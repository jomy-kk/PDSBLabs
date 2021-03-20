import numpy as np
from scipy.io.wavfile import write
import plots_aux as plt


def ex2_1_generate_sinusoid(frequency=2000, duration=3, sampling_frequency=22400, label='Signal X', show=False):
    samples = np.linspace(0, duration, int(sampling_frequency * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * samples)
    write('2k.wav', sampling_frequency, signal)
    plt.plot_signal(samples, signal, label, show=show, xlim=(0, 0.01))
    return samples, signal


def ex2_2_downsample():
    pass


def ex2_3_decimate():
    pass


def ex2_4_absolute_spectrum():
    pass


def ex2_6_interpolate():
    pass


def ex2_8_resample():
    pass


ex2_1_generate_sinusoid(show=True)
