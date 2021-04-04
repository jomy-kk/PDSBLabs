% Ex: 2
close all
clear all
%% 2)

Fs = 40000;
dt = 1/Fs;
stoptime = 3;
t = (0:dt:stoptime-dt)';
Fc = 4000;

% AM modulation

index = 0.5;
offset = 1/index;
F_am = 100;
wave4am = offset + sin(2*pi*F_am*t);

AX = ammod(wave4am, Fc,Fs,0.5);


% FM modulation

F_fm = 200;
wave4fm = sin(2*pi*F_fm*t);

F_Dev = 10*200;
FX = fmmod(wave4fm,Fc,Fs,F_Dev);

scope = dsp.SpectrumAnalyzer;
scope.SampleRate = Fs;
scope.SpectralAverages = 1;
scope.PlotAsTwoSidedSpectrum = false;
scope.RBWSource = 'Auto';
scope.PowerUnits = 'dBW';

%
scope(FX);
release(scope);

% clear('scope');
%% 2.a)


figure(1)
p_am = plot(t,wave4am)
p_am.LineWidth = 3;
ylim([0 inf])
xlim([0 0.25])
hold on
plot(t,AX)
title('AM modulated')
ylabel('Amplitude')
xlabel('Time (secs)')

figure(2)
p_fm = plot(t,wave4fm)
p_fm.LineWidth = 3;
xlim([0 0.25])
hold on
plot(t,FX)
title('FM modulated')
ylabel('Amplitude')
xlabel('Time (secs)')

%% 2.b)

figure(3)
spectrogram(AX,64,60,500,Fs,'yaxis');
title('AM modulated spectogram')
% uses hamming window - report
% change overlap - report
xlim([0 0.25])

figure(4)
spectrogram(FX,64,60,500,Fs,'yaxis');
title('FM modulated spectogram')
ylim([0 inf])
xlim([0 0.25])


%% 2.c) Extra

figure(5)

subplot(2,2,1)
p_am = plot(t,wave4am)
p_am.LineWidth = 3;
ylim([0 inf])
xlim([0 0.25])
hold on
plot(t,AX)
title('AM modulated')
ylabel('Amplitude')
xlabel('Time (secs)')


subplot(2,2,2)
spectrogram(AX,64,60,500,Fs,'yaxis');
xlim([0 0.25])
ylim([0 10])
title('AM modulated spectogram')



subplot(2,2,3)
p_fm = plot(t,wave4fm)
p_fm.LineWidth = 3;
xlim([0 0.01])
hold on
plot(t,FX)
title('FM modulated')
ylabel('Amplitude')
xlabel('Time (secs)')

subplot(2,2,4)
spectrogram(FX,64,60,500,Fs,'yaxis');
title('FM modulated spectogram')
xlim([0 0.1])
ylim([0 10])

