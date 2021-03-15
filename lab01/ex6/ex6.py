import numpy as np
import matplotlib.pyplot as plt
from pyedflib.highlevel import read_edf
from datetime import datetime, time

signals, signal_headers, header = read_edf('../data_Lab01/russek_rc_reduced.edf')
duration = 320  # seconds
timescale = 20  # seconds
start_time = time(21, 5, 30)
n_signals = len(signals)

time_ahead = (datetime.combine(header['startdate'].date(), start_time) - header['startdate']).total_seconds() - 1

fig = plt.figure(figsize=(30, 60))
fig.suptitle("Patient: " + header['patientname'] + "         Technician: " + header['technician'] + "         Start time: " + str(start_time), fontsize=16)

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

