% Ex: 2
close all
clear all
%% 2.a) 

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

mod_am = ammod(wave4am, Fc,Fs,0.5);


% FM modulation

F_fm = 200;
wave4fm = sin(2*pi*F_fm*t);

F_Dev = 10*200;
mod_fm = fmmod(wave4fm,Fc,Fs,F_Dev);

%% 2.b)
% 0.25 secs of mod_am
% adaptation - 0.01 secs of mod_fm

wave_am_4plot = wave4am(1:(1/12)*length(wave4am));
mod_am4plot = mod_am(1:(1/12)*length(mod_am));

wave_fm_4plot = wave4fm(1:(1/300)*length(wave4fm));
mod_fm4plot = mod_fm(1:(1/300)*length(mod_fm));

stoptime4plot = 0.25;
t4plot_am = (0:dt:stoptime4plot-dt)';

stoptime4plot = 0.01;
t4plot_fm = (0:dt:stoptime4plot-dt)';

figure(1)
p_am = plot(t4plot_am,wave_am_4plot)
p_am.LineWidth = 3;
ylim([0 inf]) 
hold on
plot(t4plot_am,mod_am4plot)
title('AM modulated')
ylabel('Amplitude')
xlabel('Time (secs)')

figure(2)
p_fm = plot(t4plot_fm,wave_fm_4plot)
p_fm.LineWidth = 3;
hold on
plot(t4plot_fm,mod_fm4plot)
title('FM modulated')
ylabel('Amplitude')
xlabel('Time (secs)')

%% 2.b)

figure(3)
spectrogram(mod_am,64,0,500,Fs,'yaxis');
title('AM modulated spectogram')

    
figure(4)
spectrogram(mod_fm,64,0,500,Fs,'yaxis');
title('FM modulated spectogram')
% spectrogram(x,128,120,128,fs,'yaxis')

%% 2.c) Extra

figure(5)

subplot(2,2,1)
p_am = plot(t4plot_am,wave_am_4plot)
p_am.LineWidth = 3;
ylim([0 inf]) 
hold on
plot(t4plot_am,mod_am4plot)
title('AM modulated')
ylabel('Amplitude')
xlabel('Time (secs)')

subplot(2,2,2)
spectrogram(mod_am,64,0,500,Fs,'yaxis');
xlim([0 0.25])
ylim([0 10])
title('AM modulated spectogram')



subplot(2,2,3)
p_fm = plot(t4plot_fm,wave_fm_4plot)
p_fm.LineWidth = 3;
hold on
plot(t4plot_fm,mod_fm4plot)
title('FM modulated')
ylabel('Amplitude')
xlabel('Time (secs)')

subplot(2,2,4)
spectrogram(mod_fm,64,0,500,Fs,'yaxis');
title('FM modulated spectogram')
xlim([0 0.01])
ylim([0 10])


%% 2.c)

%[stft, f, t] = stft(x, wlen, h, nfft, fs)
figure(6)
[stft_am, ~, t] = stft_prof(mod_am, 64, 500, 16, Fs);
plot(t,stft_am)

figure(7)
[stft_fm, ~, t] = stft_prof(mod_fm, 64, 500, 16, Fs);
plot(t,stft_fm)

%% 3)
%Y = chirp(T,F0,T1,F1)

Fs_chirp = 8000;
dt = 1/Fs_chirp;
stoptime_chirp = 3;
t_chirp = (0:dt:stoptime_chirp-dt)';

chirp_1 = chirp(t_chirp,400,3,2000);    
chirp_2 = chirp(t_chirp,2000,3,1000);

chirp_3 = chirp_1+chirp_2;

figure(8)
subplot(2,2,1)
spectrogram(chirp_3,256,0,500,Fs_chirp,'yaxis');
title('Window points = 256')
subplot(2,2,3)
spectrogram(chirp_3,2048,0,500,Fs_chirp,'yaxis');
title('Window points = 2048')
subplot(2,2,2)
spectrogram(chirp_3,4096,0,500,Fs_chirp,'yaxis');
title('Window points = 4096')

