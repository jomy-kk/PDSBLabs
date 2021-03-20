import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('macosx')


def plot_signal(samples, signal, label, xlim=None, ylim=None, show=False):
    fig = plt.figure(figsize=(16, 4))
    fig.tight_layout()
    plt.plot(samples, signal)
    plt.title(label)
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    fig.savefig("results/" + label + ".png", bbox_inches='tight')
    if show:
        plt.show()

