import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


# Asked procedure

def moving_average(x, n):
    y = np.zeros(len(x))
    x_new = np.zeros(len(x)+n-1)

    for k in range(n-1):  # mirroring the signal to get the first N-1 samples average
        x_new[k] = -x[n-k+1]

    for m in range(len(x)):
        x_new[m+(n-1)] = x[m]

    for i in range(len(x)):
        sum = 0
        for j in range(i, i+n-1):
            sum += x_new[j]
        avg = sum / n
        y[i] = avg

    return y


# Apply moving average

mysine = loadmat("../ex2/mysine.mat")
signal, sampling_frequency = mysine["data"][0], mysine["rate"][0][0]

y_10 = moving_average(signal, 10)
y_150 = moving_average(signal, 150)

duration = 2  # second

plt.figure(figsize=(16, 8))
ax1 = plt.subplot(3, 1, 1)
ax1.plot(signal)
ax1.axis([0, 200, -1, 1])
plt.title('Original Signal')
plt.ylabel('Amplitude')
plt.xlabel('Number of samples')

ax2 = plt.subplot(3, 1, 2)
ax2.plot(y_10)
ax2.axis([0, 200, -1, 1])
plt.title('MA with N=3')
plt.ylabel('Amplitude')
plt.xlabel('Number of samples')

ax3 = plt.subplot(3, 1, 3)
ax3.plot(y_150)
ax3.axis([0, 200, -1, 1])
plt.title('MA with N=50')
plt.ylabel('Amplitude')
plt.xlabel('Number of samples')

plt.subplots_adjust(hspace=0.5)
plt.show()

