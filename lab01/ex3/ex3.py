import numpy as np
import matplotlib.pyplot as plt


def sine(frequency, sampling_frequency, duration):
    samples = np.linspace(0, duration, int(sampling_frequency * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * samples)
    return samples, signal


def plot(signal, name=""):
    plt.figure(1)
    plt.title(name)
    plt.plot(signal[1])
    plt.show()


# Generate signal

T = 0.01
sin1 = sine(10, 1/T, 4-T)
#plot(sin1, "sin1")
sin2 = sine(6, 1/T, 4-T)
#plot(sin2, "sin2")

intervals = sin1[0]
signal = 10*sin1[1] + 40*sin2[1]
#plot((intervals, signal), "signal = sin1 + sin2")


# Asked procedure
def zero_crossing(signal):
    """
    The zero-crossing method computes where the given signal amplitude crosses zero and returns
    a list of the sample indices of signal after which a zero crossing occurs.
    For instance, if the sign of the amplitude changes between sample 3 and sample 4, it will
    return a list where 4 is included.

    :param signal: an array with the amplitude values of a signal.
    :return: the indices of signal after which a zero crossing occurs.

    Note 1: Using numpy.signbit() is a little bit quicker than numpy.sign(), since it's implementation is simpler.
    Note 2: It deals correctly with zeros in the input array. However there is one drawback: if the input array finishes
    with a zero (0) it will not detect a zero crossing there.
    """

    zeros = np.where(np.diff(np.signbit(signal)))
    return zeros[0] + 1  # +1 to return the indices AFTER which a zero-crossing occurs


def avg_period(signal, sampling_frequency, duration):
    """
    It computes the average period of a given signal, using the zero-crossing method.
    :param signal: an array with the amplitude values of a signal.
    :return (int): estimated average period of signal.

    Note: This technique is very simple and inexpensive but is not very accurate.
    It is mostly used to get a rough fundamental frequency of the signal.
    """

    zero_crossing_indexes = zero_crossing(signal)
    x = np.linspace(0, duration, int(sampling_frequency * duration), endpoint=False)
    x0 = np.empty(len(zero_crossing_indexes))  # vector of x-values for cross points

    # assume a straight line approximation between adjacent points
    for i in range(len(zero_crossing_indexes)):
        this_cross = zero_crossing_indexes[i]

        # interpolate to get x coordinate of approx 0
        x1 = this_cross - 1  # before crossing x-value
        x2 = this_cross  # after crossing x-value
        y1 = signal[x1]  # before crossing y-value
        y2 = signal[x2]  # after crossing y-value

        ratio = (0-y1) / (y2-y1)  # interpolate to find 0
        x0[i] = x[x1] + (ratio * (x[x2] - x[x1]))  # estimate of x value

        # Note: Some other interpolation based on neighboring points might be better (spline, cubic, ...)

    # Compute average period
    delta_t = np.diff(x0)  # time between each crossing
    delta_t_original_values = delta_t.copy()
    delta_t = np.around(delta_t, 3)  # three decimal places precision only
    succession = np.where(delta_t == delta_t[0])[0]  # try to find a pattern

    step = succession[1]  # step of the pattern succession
    for i in range(len(succession)):
        succession[i] -= i * step

    if np.all(succession == 0):  # this proves this was a succession u(n+1) = u(n) + step
        '''
        # check harmonics
        h = 1
        H = len(succession)
        while round(signal[zero_crossing_indexes[0]], 3) != round(signal[zero_crossing_indexes[step*h]], 3):
            print(round(signal[zero_crossing_indexes[0]], 3), round(signal[zero_crossing_indexes[step*h]], 3))
            h += 1
            if h == H:
                print("Could not find a period.")
        print(delta_t_original_values)
        return np.sum(delta_t_original_values[0:step*h])
        '''

        trimmed_delta_t_original_values = delta_t_original_values[:-(len(delta_t_original_values) % (step * 2))]
        periods = np.split(trimmed_delta_t_original_values, len(trimmed_delta_t_original_values)/(step * 2))
        return np.mean(np.sum(periods, axis=1), axis=0)

    else:
        print("Could not find a period.")
        return


# Apply method

plt.figure(figsize=(16, 8))
plt.title("Zero-crossing points of 'signal'")
plt.plot(intervals, signal, '-gD', markevery=zero_crossing(signal))
plt.plot(intervals, [0]*intervals, 'black')
plt.xlabel("seconds")
plt.ylabel("Amplitude")
plt.show()

print("Estimated period:", avg_period(signal, 1/T, 4-T), "s")





