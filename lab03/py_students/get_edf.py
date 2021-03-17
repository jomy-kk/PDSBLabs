from pyedflib.highlevel import read_edf


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

