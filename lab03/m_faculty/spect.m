function [Pw Fq] = spect (x, Fs)
L = length(x);             % Length of signal
P = abs(fft(x)/L);
Pw = P(1:floor(L/2)+1);
Pw(2:end-1) = 2*Pw(2:end-1);
Fq = Fs*(0:L/2-1)/L;
if length(Pw)> length(Fq) 
    Pw = Pw(1:length(Fq));
else
    Fq = Fq(1:length(Pw));
end