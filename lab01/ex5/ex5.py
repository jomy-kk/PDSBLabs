from sys import getrecursionlimit, setrecursionlimit
from scipy.signal import find_peaks
import numpy as np
from pyedflib.highlevel import read_edf, read_edf_header
import matplotlib.pyplot as plt

# Point 2

def show_table(data, row_headers, column_headers):
    format_row = "{:>20}" * (len(row_headers) + 1)
    print(format_row.format("", *row_headers))
    for el, row in zip(column_headers, data):
        print(format_row.format(el, *row))


signals, signal_headers, header = read_edf('../data_Lab01/russek_rc_reduced.edf')

table_data = []
channels = []

for i in range(len(signals)):
    label = signal_headers[i]['label']
    fs = signal_headers[i]['sample_rate']
    duration = len(signals[i])/fs  # cant find find any "duration" attribute in headers (?)
    # but like this, there is a "Duration" attribute, try it:
    # duration = read_edf_header('../data_Lab01/russek_rc_reduced.edf')["Duration"]
    table_data.append([label, fs, duration])
    channels.append(i + 1)

show_table(table_data, ["Label", "Sampling frequency", "Duration (s)"], channels)
print("Total number of channels:", len(channels))


# Point 3 (without get_edf procedure of the professor)

e = signals[7]  # looking in the table
e = e[:int(len(e)*200/duration) + 1]  # trimming for the first 200 seconds (inclusive)
x = np.linspace(0, duration, len(e), endpoint=False)

fig = plt.figure(figsize=(30, 8))
plt.plot(x, e)
plt.xlim([0, 20])  # only the first 20 seconds
plt.title(signal_headers[7]["label"])
plt.ylabel('Amplitude [' + signal_headers[7]['dimension'] + ']')
plt.xlabel('seconds')
fig.tight_layout()
#plt.show()


# Point 4 (using get_edf procedure of the professor)

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


f, sf = get_edf('../data_Lab01/russek_rc_reduced.edf', 'ECG2')
f = f[:int(len(e)*200/duration) + 1]  # trimming for the first 200 seconds (inclusive)
x = np.linspace(0, duration, len(f), endpoint=False)

fig = plt.figure(figsize=(30, 8))
plt.plot(x, f)
plt.xlim([0, 20])  # only the first 20 seconds
plt.title('ECG2')
plt.ylabel('Amplitude [uV]')
plt.xlabel('seconds')
fig.tight_layout()
#plt.show()


# Point 5a (using professor find_extremas function)

def find_extremas(h):
    lh = len(h)
    df = np.empty(lh)
    maximas = []
    minimas = []
    for j in range(1, lh - 1):
        df[j] = h[j] - h[j - 1]
        df[j + 1] = h[j + 1] - h[j]

        if df[j] == 0 or (df[j] >= 0 and df[j + 1] < 0):
            maximas.append([j, h[j]])
        else:
            if df[j] == 0 or (df[j] <= 0 and df[j + 1] > 0):
                minimas.append([j, h[j]])

    # maximas = [1, h(1);maximas;lh, h(lh)];
    # minimas = [1, h(1); minimas; lh, h(lh)];

    st = [0, h[0]]
    ed = [lh-1, h[lh-1]]

    return st, maximas, minimas, ed


recursion_limit = getrecursionlimit()
setrecursionlimit(700000)
maxima = (find_extremas(e)[1])
setrecursionlimit(recursion_limit)

x = np.linspace(0, duration, len(e), endpoint=False)
fig = plt.figure(figsize=(30, 8))
plt.plot(x, e)
#plt.scatter(maxima)
plt.xlim([0, 60])  # only the first 60 seconds
plt.title("ECG1 with maxima highlighted")
plt.ylabel('Amplitude [uV]')
plt.xlabel('seconds')
fig.tight_layout()
#plt.show()


# Point 5b (using scipy.signal.find_peaks)

maxima_indices = find_peaks(e, prominence=1000)[0]
fig = plt.figure(figsize=(30, 8))
plt.plot(x, e, '-gD', markevery=maxima_indices)
plt.xlim([0, 60])  # only the first 60 seconds
plt.title("ECG1 with peaks highlighted")
plt.ylabel('Amplitude [uV]')
plt.xlabel('seconds')
fig.tight_layout()
#fig.show()

# Plot RRI interval (in milliseconds)
rri = np.diff(maxima_indices)/sf * 1000

fig = plt.figure(figsize=(16, 8))
plt.plot((maxima_indices/sf)[1:], rri)
plt.plot([np.mean(rri)]*len(rri), 'r--', label="Average RRI")
plt.xlim([0, 60])  # only the first 60 seconds
plt.title("ECG1 RRI")
plt.ylabel('R-R interval (ms)')
plt.xlabel('signal time (s)')
plt.legend(loc="upper right")
fig.tight_layout()
#fig.show()


# Point 6

g, sf = get_edf('../data_Lab01/russek_rc_reduced.edf', 'CINTA_TORACICA T')
g = g[:int(len(e)*120/duration) + 1]  # trimming for the first 120 seconds (inclusive)
x = np.linspace(0, duration, len(g), endpoint=False)

fig = plt.figure(figsize=(16, 8))
plt.plot(x, g)
# plt.plot([np.mean(rri)]*len(rri), 'r--', label="Average RRI")
#plt.xlim([0, 60])  # only the first 60 seconds
plt.title("CINTA_TORACICA T")
plt.ylabel('Amplitude [uV]')
plt.xlabel('seconds')
plt.legend(loc="upper right")
fig.tight_layout()
fig.show()


