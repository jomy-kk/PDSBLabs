import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyedflib.highlevel import read_edf
from scipy.io import loadmat

mpl.use('macosx')


def plot_signal(samples, signal, label, xlim=None, ylim=None, show=False):
    fig = plt.figure(figsize=(16, 4))
    fig.tight_layout()
    plt.plot(samples, signal)
    plt.title(label)
    plt.ylabel('Amplitude [uV]')
    plt.xlabel('Time [s]')
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    fig.savefig("results/" + label + ".png", bbox_inches='tight')
    if show:
        plt.show()
    plt.show()


def plot_signal_processed(original_ts, original_signal, processed_ts, processed_signal, processing_label, label, xlim=None, ylim=None, show=False):
    fig = plt.figure(figsize=(16, 4))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    plt.title('Original')
    plt.plot(original_ts, original_signal, 'b')
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)

    plt.subplot(2, 1, 2)
    plt.title(processing_label)
    plt.plot(processed_ts, processed_signal, 'r')
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)

    fig.savefig("results/" + label + '_' + str(xlim) + ".png", bbox_inches='tight')
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



def get_edf(fname, channel):
    signals, signal_headers, header = read_edf(fname)
    chn = None
    for h in range(len(signal_headers)):
        if signal_headers[h]['label'] == channel:
            chn = h
    if chn is None:
        raise ValueError(channel + " cannot be found in the given EDF.")
    else:
        sf = signal_headers[chn]['sample_rate']
        b = signals[chn]
        return b, sf


def get_signals():
    s16, sf_s16 = get_edf('../data/s16.edf', 'C3-A2')

    # start times in seconds
    sA_start_time = 5*60 + 52
    sB_start_time = 8*60 + 3
    sC_start_time = 16*60

    n_samples = 4096

    sA = s16[sA_start_time*sf_s16:sA_start_time*sf_s16 + n_samples]
    sB = s16[sB_start_time*sf_s16:sB_start_time*sf_s16 + n_samples]
    sC = s16[sC_start_time*sf_s16:sC_start_time*sf_s16 + n_samples]

    ecg_signal = loadmat('../data/ecg.mat')['ecg']
    sf_ecg = 128  # Hz

    return (sA, sf_s16), (sB, sf_s16), (sC, sf_s16), (ecg_signal, sf_ecg)


get_signals()















