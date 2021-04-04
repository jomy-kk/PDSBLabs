import numpy as np
from scipy.io.wavfile import read
from scipy.signal import get_window, find_peaks

import plots_aux as plt
from lab04.ex1.spect import spect


def ex4(filepath, window_types, wlens, overlaps, xlim=[3, 4], label="Signal X", show=False):
    sf, signal = read(filepath)
    t0, t1 = sf * xlim[0], sf * xlim[1] + 1

    x = signal[t0:t1]
    ts = np.linspace(3, 4, int(sf) + 1, endpoint=True)
    print("# Samples of X:", sf)

    plt.plot_signal(ts, x, label, xlim=xlim, show=show)

    pw, fq = spect(x, sf)
    plt.plot_absolute_spectrum(fq, pw, label + " Absolute Spectrum", show=show)
    peaks_indexes = find_peaks(pw, prominence=500)[0]
    peaks_freqs = np.array(fq)[list(peaks_indexes)]
    print(peaks_freqs)

    for window_type in window_types:
        for wlen in wlens:
            for overlap in overlaps:
                overlap = int(overlap * wlen)
                plt.plot_spectogram(x, sf, get_window(window_type, wlen), wlen, overlap,
                                    label="Tukey window (0.25), overlap " + str(overlap),
                                    show=show)
                # + (str(window_type))[0].upper() + str(window_type)[1:] +


filepath = "../data/looneyTunes.wav"
ex4(filepath, [('tukey', 0.25)], [256, ], [0.1, 0.3, 0.5, 0.8, 0.9, 0.99])
