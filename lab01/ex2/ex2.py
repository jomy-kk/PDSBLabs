import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.io import savemat


def plot(signal, name=""):
    plt.figure(figsize=(16,8))
    plt.title(name)
    plt.plot(signal[0], signal[1])
    plt.show()


def sine(frequency, sampling_frequency, duration):
    samples = np.linspace(0, duration, int(sampling_frequency * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * samples)
    return samples, signal


signal_200 = sine(200, 22000, 1)
signal_3000 = sine(3000, 22000, 1)
#plot(signal_200)
#plot(signal_3000)

signal_888 = sine(888, 900, 2)
write("mysine.wav", 22000, signal_888[1])
savemat("mysine.mat", {"rate": 22000, "data": signal_888[1]})
plot(signal_888, "Original")

