from scipy.io.wavfile import read
import numpy as np
from numpy import*
import matplotlib.pyplot as plt
from playsound import playsound


def read_signal(filepath):
    signal = read(filepath)
    return signal

def metadata(signal):
    print ("Sampling rate (Hz)", signal[0])
    print ("Duration (s)", len(signal[1]) / signal[0])


def plot(signal, name=""):
    plt.figure(1)
    plt.title(name)
    plt.plot(signal[1])
    plt.show()

# signal.wav
signal = read_signal("../data_Lab01/signal.wav")
metadata(signal)
plot(signal)

# tunes.wav
tunes = read_signal("../data_Lab01/tunes.wav")
metadata(tunes)
plot(tunes)

# small envelope of tunes.wav
plt.figure(1)
plt.title("Tunes cut")
plt.plot(linspace(1000, 1000+tunes[0], tunes[0]), tunes[1][1000:1000+tunes[0]])
plt.show()

