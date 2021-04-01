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


def plot_absolute_spectrum(fq, pw, label, xmax=None, xticks=None, ylim=None, show=False):
    fig = plt.figure(figsize=(16, 4))
    fig.tight_layout()
    plt.plot(fq, pw)
    plt.title(label)
    plt.ylabel('Power')
    plt.xlabel('Frequency [Hz]')
    if ylim is not None:
        plt.ylim(ylim)
    if xticks is not None:
        plt.xticks([0] + xticks + [int(fq[-1])])
    if xmax is not None:
        plt.xlim(0, xmax)
    else:
        plt.xlim(0, fq[-1])
    fig.savefig("results/" + label + ".png", bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_spectogram(signal, sf, window, wlen, noverlap, label, show=False):
    plt.clf()
    fig = plt.figure(figsize=(16, 6))

    s, f, t, im = plt.specgram(signal, NFFT=wlen, Fs=sf, window=window, noverlap=noverlap, mode='magnitude')
    # mode='magnitude' to match the MATLAB function which returns the STFT, otherwise it would return the PSD

    plt.title(label)
    plt.ylim((0, 300))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.colorbar(label='Power / frequency [dB/Hz]')
    fig.savefig("results/" + label + ".png", bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
