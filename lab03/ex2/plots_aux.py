import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('macosx')


def plot_signal_processed(original_ts, original_signal, processed_ts, processed_signal, processing_label, label, xlim=None, ylim=None, show=False):
    fig = plt.figure(figsize=(16, 4))
    fig.tight_layout()
    plt.plot(original_ts, original_signal, '--r', label='Original')
    plt.plot(processed_ts, processed_signal, label=processing_label)
    plt.title(label)
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend(loc='upper right')
    fig.savefig("results/" + label + ".png", bbox_inches='tight')
    if show:
        plt.show()


