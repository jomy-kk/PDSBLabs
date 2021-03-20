import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, resample, butter, lfilter
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
               'results_findpeaks', (-150, 250))
    plot_peak_intervals(ecg_peak_times, ecg_label, 'findpeaks', 'RRI', 'results_findpeaks')

    # Try to resolve better [2100:] s interval by decreasing prominence
    t0 = 2100 * ecg_sf  # in #sample
    ecg_peak_indices_second_part, _ = find_peaks(filtered_ecg_signal[t0:], prominence=50, distance=150,
                                                 width=(None, 280))
    ecg_peak_indices_second_part += t0
    ecg_peak_times_second_part = ecg_peak_indices_second_part / ecg_sf

    plot_peaks(filtered_ecg_signal, ecg_times, np.concatenate((ecg_peak_times[:t0], ecg_peak_times_second_part)),
               np.concatenate((ecg_peak_indices[:t0], ecg_peak_indices_second_part)),
               ecg_envelopes, ecg_label, 'findpeaks_corrected', 'results_findpeaks', (-150, 250))
    plot_peak_intervals(np.concatenate((ecg_peak_times[:t0], ecg_peak_times_second_part)), ecg_label,
                        'findpeaks_corrected', 'RRI', 'results_findpeaks', ylim=(0, 2))


# Option 2 - Using biosppy.signal.ecg functions
# There are many algorithms; let's try some to see if they resolve better [105, 125]s issue
def ex1_2_1_biosppy():
    filtered_ecg_signal = ecg_processing.ecg(ecg_signal, ecg_sf, show=False)['filtered']

    ecg_peak_indices = ecg_processing.hamilton_segmenter(ecg_signal, ecg_sf)['rpeaks']
    ecg_peak_times = ecg_peak_indices / ecg_sf

    plot_peaks(filtered_ecg_signal, ecg_times, ecg_peak_times, ecg_peak_indices, ecg_envelopes, ecg_label, 'hamilton', 'results_hamilton',
               (-150, 250))
    plot_peak_intervals(ecg_peak_times, ecg_label, 'hamilton', 'RRI', 'results_hamilton')

    """
    ecg_peak_indices = ecg_processing.christov_segmenter(ecg_signal, ecg_sf)
    engzee_ecg_peak_indices = ecg_processing.engzee_segmenter(ecg_signal, ecg_sf)
    gamboa_ecg_peak_indices = ecg_processing.gamboa_segmenter(ecg_signal, ecg_sf)
    hamilton_ecg_peak_indices = ecg_processing.hamilton_segmenter(ecg_signal, ecg_sf)
    ssf_ecg_peak_indices = ecg_processing.ssf_segmenter(ecg_signal, ecg_sf, threshold=50, before=0.5, after=0.7)
    """


# 1.2.2. Plot P-P interval of PLETH
ppg_envelopes = ((395, 420), (545, 575), (2620, 2640), (3200, 3220))


def ex1_2_2_scipy():
    # Find peaks
    print(ppg_sf)
    filtered_ppg_signal = ppg_processing.bvp(ppg_signal, ppg_sf, show=False)['filtered']
    ppg_peak_indices, _ = find_peaks(filtered_ppg_signal, prominence=35, distance=0.5 * ppg_sf)
    ppg_peak_times = ppg_peak_indices / ppg_sf

    plot_peaks(filtered_ppg_signal, ppg_times, ppg_peak_times, ppg_peak_indices, ppg_envelopes, ppg_label, 'findpeaks',
               'results_findpeaks', (-300, 300))
    plot_peak_intervals(ppg_peak_times, ppg_label, 'findpeaks', 'PPI', 'results_findpeaks')


def ex1_2_2_biosspy():
    # Find onsets
    # It uses Zong et al. approach, which skips corrupted signal parts
    result = ppg_processing.bvp(ppg_signal, ppg_sf, show=False)
    filtered_ppg_signal, ppg_onset_indices, heart_rate_ts, heart_rate = result['filtered'], result['onsets'], result[
        'heart_rate_ts'], result['heart_rate']
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


def ex1_2_3_mean_square_error():
    '''"# Empirically, hamilton segmenter performed better for RRI
    ecg_signal_resampled = resample(ecg_signal,
                                    len(ppg_signal))  # we need to resample the eeg signal to rate of the ppg signal
    assert len(ecg_signal_resampled) == len(ppg_signal)
    ecg_peak_indices = ecg_processing.hamilton_segmenter(ecg_signal_resampled, ppg_sf)[
        'rpeaks']  # now it has the same sf as the ppg signal
    ecg_peak_times = ecg_peak_indices / ppg_sf
    rri = np.diff(ecg_peak_times)
    '''

    # Empirically, Elgendi performed better for RRI
    ecg_signal_resampled = resample(ecg_signal, len(ppg_signal))  # we need to resample the eeg signal to rate of the ppg signal
    assert len(ecg_signal_resampled) == len(ppg_signal)
    ecg_sf = ppg_sf
    ecg_peak_times = ex1_2_4_elgendi()
    rri = np.diff(ecg_peak_times)

    # Empirically, find_peaks performed better for PPI
    filtered_ppg_signal = ppg_processing.bvp(ppg_signal, ppg_sf, show=False)['filtered']
    ppg_peak_indices, _ = find_peaks(filtered_ppg_signal, prominence=35, distance=0.5 * ppg_sf)
    ppg_peak_times = ppg_peak_indices / ppg_sf
    ppi = np.diff(ppg_peak_times)

    # we need to trim the bigger one, if the algorithms did not find the same amount of peaks
    if len(rri) > len(ppi):
        rri = rri[:len(ppi)]
    if len(ppi) > len(rri):
        ppi = ppi[:len(rri)]

    mse = np.square(np.subtract(rri, ppi)).mean()
    print("MSE:", mse)


def ex1_2_4_elgendi():
    """
    Elgendi et al.[2010] uses a bandpass filter and two moving averages to segment the ECG signal into blocks containing
    potential QRS complexes. The ECG signal is filtered using a second order Butterworth IIR filter with a passband of
    8 – 20 Hz. The first moving average has a window of 120 ms to match the approximate duration of a QRS complex.
    A wider window of 600 ms is used for the second moving average to match the approximate duration of a complete
    heartbeat. Both moving averages are performed on the rectified bandpass filtered signal. Sections of the filtered
    ECG where the amplitude of the first moving average is higher than that of the second are marked as blocks
    containing a potential heartbeat. Blocks with a width of less than 80 ms are ignored as this is smaller than a QRS
    complex. The maximum value of the filtered ECG in each block is then stored as a detected QRS. Detections which
    follow the previous one by less than 300 ms are removed.

    :return:
    ecg_peak_times: array_like
    Time stamps in seconds where the R-peaks occur in the given signal.

    Copied and adapted from:
    Copyright (C) 2019 Luis Howell & Bernd Porr
    under the GPL GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
    See more: https://github.com/berndporr/py-ecg-detectors/blob/master/ecgdetectors.py
    """

    def MWA_cumulative(input_array, window_size):

        ret = np.cumsum(input_array, dtype=float)
        ret[window_size:] = ret[window_size:] - ret[:-window_size]

        for i in range(1, window_size):
            ret[i - 1] = ret[i - 1] / i
        ret[window_size - 1:] = ret[window_size - 1:] / window_size

        return ret

    f1, f2 = 8 / ecg_sf, 20 / ecg_sf
    b, a = butter(2, [f1 * 2, f2 * 2], btype='bandpass')

    filtered_ecg = lfilter(b, a, ecg_signal)

    window1 = int(0.12 * ecg_sf)
    mwa_qrs = MWA_cumulative(abs(filtered_ecg), window1)

    window2 = int(0.6 * ecg_sf)
    mwa_beat = MWA_cumulative(abs(filtered_ecg), window2)

    blocks = np.zeros(len(ecg_signal))
    block_height = np.max(filtered_ecg)

    for i in range(len(mwa_qrs)):
        if mwa_qrs[i] > mwa_beat[i]:
            blocks[i] = block_height
        else:
            blocks[i] = 0

    QRS = []

    for i in range(1, len(blocks)):
        if blocks[i - 1] == 0 and blocks[i] == block_height:
            start = i

        elif blocks[i - 1] == block_height and blocks[i] == 0:
            end = i - 1

            if end - start > int(0.08 * ecg_sf):
                detection = np.argmax(filtered_ecg[start:end + 1]) + start
                if QRS:
                    if detection - QRS[-1] > int(0.3 * ecg_sf):
                        QRS.append(detection)
                else:
                    QRS.append(detection)

    ecg_peak_indices = np.array(QRS)
    ecg_peak_times = ecg_peak_indices / ecg_sf  # in seconds

    plot_peaks(filtered_ecg, ecg_times, ecg_peak_times, ecg_peak_indices, ecg_envelopes, ecg_label, 'elgendi', 'results_elgendi', (-150, 150))
    plot_peak_intervals(ecg_peak_times, ecg_label, 'elgendi', 'RRI', 'results_elgendi')

    return ecg_peak_times


# ex1_2_1_scipy()
# ex1_2_1_biosppy()
#ex1_2_2_scipy()
ex1_2_2_biosspy()
# ex1_2_3_mean_square_error()
# ex1_2_4_elgendi()

