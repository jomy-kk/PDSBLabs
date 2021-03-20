import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('macosx')


def plot_peak_intervals(peak_times, label, method, interval_label, directory_to_save, ylim=None):
    # Compute interval
    intervals = np.diff(peak_times)  # in seconds
    fig = plt.figure(figsize=(16, 2.5))
    fig.tight_layout()
    plt.plot(peak_times[1:], intervals, '.')
    mean = np.mean(intervals)
    plt.plot((0, peak_times[-1]), (mean, mean), 'r-', label='Mean')
    plt.title(label)
    plt.ylabel(interval_label + ' [s]')
    plt.xlabel('Signal timestamps [s]')
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend(loc='upper right')
    fig.savefig(directory_to_save + "/" + label + "_" + interval_label + '_' + method + ".png", bbox_inches='tight')

    # Interval Statistics
    print(interval_label, "mean:", mean)
    print(interval_label, "standard deviation:", np.std(intervals))


def plot_peaks(signal, signal_times, peak_times, peak_indices, envelopes, label, method, directory_to_save, ylim=None):
    fig = plt.figure(figsize=(16, 2.5))
    fig.tight_layout()
    plt.plot(signal_times, signal)
    plt.plot(peak_times, signal[peak_indices], 'ro')
    plt.title(label)
    plt.ylabel('Amplitude [uV]')
    plt.xlabel('Time [s]')
    if ylim is not None:
        plt.ylim(ylim)
    for env in envelopes:
        plt.xlim(env)
        fig.savefig(directory_to_save + "/" + label + "_peaks_" + method + '_' + str(env) + ".png", bbox_inches='tight')


def plot_onsets(signal, signal_times, onset_times, envelopes, label, method, directory_to_save, ylim=None):
    fig = plt.figure(figsize=(16, 2.5))
    fig.tight_layout()
    plt.plot(signal_times, signal)
    plt.vlines(onset_times, ylim[0] if ylim is not None else -100, ylim[1] if ylim is not None else 100,
               color='m',)
    plt.title(label)
    plt.ylabel('Amplitude [uV]')
    plt.xlabel('Time [s]')
    if ylim is not None:
        plt.ylim(ylim)
    for env in envelopes:
        plt.xlim(env)
        fig.savefig(directory_to_save + "/" + label + "_peaks_" + method + '_' + str(env) + ".png", bbox_inches='tight')

