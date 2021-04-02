import numpy as np
from scipy.io.wavfile import read
from scipy.signal import get_window, find_peaks

import plots_aux as plt
from lab04.ex1.spect import spect

filepath = "../data/looneyTunes.wav"



def ex4(label="Signal X", xticks=[3,4], show=False):
    sf, signal = read(filepath)
    t0, t1 = sf*xticks[0], sf*xticks[1] + 1

    x = signal[t0:t1]
    ts = np.linspace(3, 4, int(sf) + 1, endpoint=True)
    print("# Samples of X:", sf)

    plt.plot_signal(ts, x, label, xlim=xticks, show=show)

    pw, fq = spect(x, sf)
    plt.plot_absolute_spectrum(fq, pw, label + " Absolute Spectrum", show=show)
    peaks_indexes = find_peaks(pw, prominence=500)[0]
    peaks_freqs = np.array(fq)[list(peaks_indexes)]
    print(peaks_freqs)

    # Parameters
    window_types = [('tukey', 0.25),]
    wlens = [256, ]  # 256
    overlaps = [0.9, ]

    for window_type in window_types:
        for wlen in wlens:
            for overlap in overlaps:
                overlap = int(overlap * wlen)
                plt.plot_spectogram(x, sf, get_window(window_type, wlen), wlen, overlap,
                        label=label + " Spectrogram " + str(window_type) + "_" + str(wlen) + "_" + str(overlap),
                        show=show)


ex4()