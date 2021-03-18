import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import biosppy.signals.bvp as ppg_processing
import biosppy.signals.ecg as ecg_processing
from lab03.py_students.get_edf import get_edf
mpl.use('macosx')

# Data collection
# Note: Doesn't read all the channels for simplicity.
file_path = '../data/plm3_r_ECG_PPG_1h.edf'
ecg_label, ppg_label = 'ECG1-ECG2', 'PLETH'
ecg_signal, ecg_sf = get_edf(file_path, ecg_label)
ppg_signal, ppg_sf = get_edf(file_path, ppg_label)

# The signals come vertically mirrored (i.e. inverted amplitudes)
ecg_signal, ppg_signal = -ecg_signal, -ppg_signal
ecg_times = np.linspace(0, len(ecg_signal) / ecg_sf, len(ecg_signal))
ppg_times = np.linspace(0, len(ppg_signal) / ppg_sf, len(ppg_signal))


# 1.1. Plot both signals in the interval [6, 306] second
def ex1_1():
    fig = plt.figure(figsize=(16, 8))
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(2, 1, 1)
    plt.plot(ecg_times, ecg_signal)
    plt.title(ecg_label)
    plt.xlim([6, 306])
    plt.ylabel('Amplitude [uV]')
    plt.xlabel('Time [s]')

    plt.subplot(2, 1, 2)
    plt.plot(ppg_times, ppg_signal)
    plt.title(ppg_label)
    plt.xlim([6, 306])
    plt.ylabel('Amplitude [uV]')
    plt.xlabel('Time [s]')

    plt.show()
    fig.savefig("both_5minute_envelope.png", bbox_inches='tight')


# 1.2.1. Plot R-R interval of ECG
envelope1 = (30, 40)
envelope2 = (105, 125)
envelope3 = (240, 260)
envelope4 = (3200, 3220)
ecg_envelopes = (envelope1, envelope2, envelope3, envelope4)
from plots_aux import plot_peak_intervals, plot_peaks, plot_onsets


# Option 1 - Using scipy.signal.find_peaks function
def ex1_2_1_scipy():
    # Find peaks
    filtered_ecg_signal = ecg_processing.ecg(ecg_signal, ecg_sf, show=False)['filtered']
    ecg_peak_indices, _ = find_peaks(filtered_ecg_signal, prominence=120, distance=150, width=(None, 280))
    ecg_peak_times = ecg_peak_indices / ecg_sf

    plot_peaks(filtered_ecg_signal, ecg_times, ecg_peak_times, ecg_peak_indices, ecg_envelopes, ecg_label, 'findpeaks',
               'results_findpeaks', (-600, 600))
    plot_peak_intervals(ecg_peak_times, ecg_label, 'findpeaks', 'RRI', 'results_findpeaks')

    # Try to resolve better [2100:] s interval by decreasing prominence
    t0 = 2100 * ecg_sf  # in #sample
    ecg_peak_indices_second_part, _ = find_peaks(filtered_ecg_signal[t0:], prominence=50, distance=150, width=(None, 280))
    ecg_peak_indices_second_part += t0
    ecg_peak_times_second_part = ecg_peak_indices_second_part / ecg_sf

    plot_peaks(filtered_ecg_signal, ecg_times, np.concatenate((ecg_peak_times[:t0], ecg_peak_times_second_part)),
               np.concatenate((ecg_peak_indices[:t0], ecg_peak_indices_second_part)),
               ecg_envelopes, ecg_label, 'findpeaks_corrected', 'results_findpeaks', (-600, 600))
    plot_peak_intervals(np.concatenate((ecg_peak_times[:t0], ecg_peak_times_second_part)), ecg_label, 'findpeaks_corrected', 'RRI', 'results_findpeaks', ylim=(0, 2))

# Option 2 - Using biosppy.signal.ecg functions
# There are many algorithms; let's try some to see if they resolve better [105, 125]s issue
def ex1_2_1_biosppy():
    ecg_peak_indices = ecg_processing.ssf_segmenter(ecg_signal, ecg_sf)['rpeaks']
    ecg_peak_times = ecg_peak_indices / ecg_sf

    plot_peaks(ecg_signal, ecg_times, ecg_peak_times, ecg_peak_indices, ecg_envelopes, ecg_label, 'ssf', 'results_ssf',
               (-600, 600))
    plot_peak_intervals(ecg_peak_times, ecg_label, 'ssf', 'RRI', 'results_ssf')

    """
    ecg_peak_indices = ecg_processing.christov_segmenter(ecg_signal, ecg_sf)
    engzee_ecg_peak_indices = ecg_processing.engzee_segmenter(ecg_signal, ecg_sf)
    gamboa_ecg_peak_indices = ecg_processing.gamboa_segmenter(ecg_signal, ecg_sf)
    hamilton_ecg_peak_indices = ecg_processing.hamilton_segmenter(ecg_signal, ecg_sf)
    ssf_ecg_peak_indices = ecg_processing.ssf_segmenter(ecg_signal, ecg_sf)
    """


# 1.2.2. Plot P-P interval of PLETH
ppg_envelopes = ((395, 420), (545, 575), (2620, 2640), (3200, 3220))


def ex1_2_2_scipy():
    # Find peaks
    filtered_ppg_signal = ppg_processing.bvp(ppg_signal, ppg_sf, show=False)['filtered']
    ppg_peak_indices, _ = find_peaks(filtered_ppg_signal, prominence=35, distance=0.5 * ppg_sf)
    ppg_peak_times = ppg_peak_indices / ppg_sf

    plot_peaks(filtered_ppg_signal, ppg_times, ppg_peak_times, ppg_peak_indices, ppg_envelopes, ppg_label, 'findpeaks',
               'results_findpeaks', (-300, 300))
    plot_peak_intervals(ppg_peak_times, ppg_label, 'findpeaks', 'PPI', 'results_findpeaks')


def ex1_2_2_biosspy():
    # Find onsets
    result = ppg_processing.bvp(ppg_signal, ppg_sf, show=False)
    filtered_ppg_signal, ppg_onset_indices, heart_rate_ts, heart_rate = result['filtered'], result['onsets'], result['heart_rate_ts'], result['heart_rate']
    ppg_onset_times = ppg_onset_indices / ppg_sf

    plot_onsets(filtered_ppg_signal, ppg_times, ppg_onset_times, ppg_envelopes, ppg_label, 'bvp',
               'results_bvp', (-300, 300))
    plot_peak_intervals(ppg_onset_times, ppg_label, 'bvp', 'Onset intervals', 'results_bvp')

    # Plot heart rate, just for fun
    fig = plt.figure(figsize=(16, 4))
    plt.plot(heart_rate_ts, heart_rate)
    mean = np.mean(heart_rate)
    plt.plot((0, heart_rate_ts[-1]), [mean, mean], 'r-', label='Mean')
    plt.xlabel('Time [s]')
    plt.ylabel('Instantaneous Heart Rate [bpm]')
    plt.grid()
    plt.title("Heart rate derived from PPG")
    fig.tight_layout()
    plt.legend(loc='upper right')
    fig.savefig("results_bvp/heart_rate")




#ex1_2_1_scipy()
#ex1_2_1_biosppy()
#ex1_2_2_scipy()
ex1_2_2_biosspy()
