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


def plot_spectogram(f, t, spect, label, show=False):
    fig = plt.figure(figsize=(10, 10))
    plt.pcolormesh(t, f, spect, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title(label)
    plt.colorbar(label='Power / frequency [dB/Hz]')
    fig.savefig("results/" + label + ".png", bbox_inches='tight')
    if show:
        plt.show()
