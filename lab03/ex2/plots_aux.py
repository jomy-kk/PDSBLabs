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


def plot_spectogram(f, t, spect, label, show=False):
    fig = plt.figure(figsize=(16, 16))
    plt.pcolormesh(t, f, spect, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title(label)
    plt.colorbar(label='Power / frequency [dB/Hz]')
    fig.savefig("results/spectograms/" + label + ".png", bbox_inches='tight')
    if show:
        plt.show()


def plot_absolute_spectrum(fq, pw, label, xmax=None, ylim=None, show=False):
    fig = plt.figure(figsize=(16, 4))
    fig.tight_layout()
    plt.plot(fq, pw)
    plt.title(label)
    plt.ylabel('Power')
    plt.xlabel('Frequency [Hz]')
    if ylim is not None:
        plt.ylim(ylim)
    if xmax is not None:
        plt.xlim(0, xmax)
    else:
        plt.xlim(0, fq[-1])
    fig.savefig("results/spectra/" + label + ".png", bbox_inches='tight')
    if show:
        plt.show()