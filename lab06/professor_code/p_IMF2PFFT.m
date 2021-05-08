% FF spectrum of matrix signals as IMFs
function PFFT = IMF2PFFT(IMF,Fs)

[N L] = size(IMF);
if (nargin == 1) Fs = 256; end  % Sampling frequency                    
T = 1/Fs;                       % Sampling period       
t = (0:L-1)*T;                  % Time vector
f = Fs*(0:(L/2))/L;
%
%plot(1000*t(1:50),IMF(1,1:50))
%title('IMF1')
%xlabel('t (milliseconds)')
%ylabel('IMF(t)');
%
figure;
title('Single-Sided Amplitude Spectrum of IMF(t) - |AS FFT(f)|');
for i = 1:N
    Y = fft(IMF(i,:));
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    subplot(N,1,i);
    plot(f,P1);
    ylabel (['i.' num2str(i)]);
    PFFT(i,:) = P1;
end
%set(gca,'xtick',[],'FontSize',8,'XLim',[0 L]);
xlabel('f (Hz)');
return