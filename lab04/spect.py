from scipy.fft import fft
from numpy import arange
from math import floor


def spect(signal, sf):
    l = len(signal)
    p = abs(fft(signal) / l)
    pw = p[1:floor(l / 2) + 1]
    pw[2: - 1] = 2 * pw[2: - 1]
    fq = sf * arange(l/2-1) / l
    if len(pw) > len(fq):
        pw = pw[:len(fq)]
    else:
        fq = fq[:len(pw)]
    return pw, fq

