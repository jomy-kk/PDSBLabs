import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from pyedflib.highlevel import read_edf
from datetime import datetime, time
from scipy.signal import iirnotch, filtfilt, freqz

signals, signal_headers, header = read_edf('../data_Lab01/russek_rc_reduced.edf')
duration = 320  # seconds
timescale = 20  # seconds
start_time = time(21, 5, 30)
n_signals = len(signals)

time_ahead = (datetime.combine(header['startdate'].date(), start_time) - header['startdate']).total_seconds() - 1

fig = plt.figure(figsize=(30, 60))
fig.suptitle("Patient: " + header['patientname'] + "         Technician: " + header['technician'] + "         Start time: " + str(start_time), fontsize=16)
'''
for i, header, signal in zip(range(1, n_signals + 1), signal_headers, signals):
    samples_ahead = int(time_ahead * header['sample_rate'])
    signal = signal[samples_ahead:int(len(signal) * timescale / duration) + samples_ahead + 1]  # trimming for the first 20 seconds (inclusive)
    x = np.linspace(0, timescale, len(signal), endpoint=False)
    ax1 = plt.subplot(n_signals, 1, i)
    ax1.plot(x, signal)
    plt.xlim([0, timescale])
    plt.title(header['label'])
    plt.ylabel('Amplitude [' + header['dimension'] + ']')
    plt.xlabel('Time [s]')

plt.subplots_adjust(hspace=0.6)
plt.show()
fig.savefig("sc1.pdf", bbox_inches='tight')
'''

# Point 5

def get_edf(fname, channel):
    signals, signal_headers, header = read_edf(fname)
    chn = 0
    for h in range(len(signal_headers)):
        if signal_headers[h]['label'] == channel:
            chn = h
    if chn == 0:
        raise ValueError(channel + " cannot be found in the given EDF.")
    else:
        sf = signal_headers[chn]['sample_rate']
        b = signals[chn]
        return b, sf


#fig = plt.figure(figsize=(30, 60))
#fig.suptitle("Patient: " + header['patientname'] + "         Technician: " + header['technician'] + "         Start time: " + str(start_time), fontsize=16)
n_signals = 6

for i, channel in zip(range(1, n_signals+1), ('EEG C3', 'EEG C4', 'EMGQ', 'EOGE', 'ECG1', 'ECG2')):
    signal, sf = get_edf('../data_Lab01/russek_rc_reduced.edf', 'ECG2')
    b, a = iirnotch(50/sf, 35)
    '''
    w, h = freqz(b, a)

    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')
    ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [rad/sample]')
    ax1.grid()
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid()
    ax2.axis('tight')
    nticks = 8
    ax1.yaxis.set_major_locator(ticker.LinearLocator(nticks))
    ax2.yaxis.set_major_locator(ticker.LinearLocator(nticks))

    plt.show()
    fig.savefig('notch_filter.png', bbox_inches='tight')
    exit(0)
    '''

    samples_ahead = int(time_ahead * sf)
    signal = signal[samples_ahead:int(len(signal) * timescale / duration) + samples_ahead + 1]  # trimming for the first 20 seconds (inclusive)
    filtered_signal = filtfilt(b, a, signal)

    x = np.linspace(0, timescale, len(filtered_signal), endpoint=False)
    ax1 = plt.subplot(n_signals, 1, i)
    ax1.plot(x, filtered_signal)
    plt.xlim([0, timescale])
    plt.title(channel)
    plt.ylabel('Amplitude [uV]')
    plt.xlabel('Time [s]')

plt.subplots_adjust(hspace=0.6)
plt.show()
fig.savefig("sc2.pdf", bbox_inches='tight')


