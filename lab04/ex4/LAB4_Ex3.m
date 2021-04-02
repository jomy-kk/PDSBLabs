%% 3)
%Y = chirp(T,F0,T1,F1)

Fs_chirp = 8000;
dt = 1/Fs_chirp;
st_chirp = 3;
t_chirp = (0:dt:st_chirp-dt)';

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