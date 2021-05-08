% Lab 6

import 

%% Exercise a)

% Load data
load('data/ecg.mat')
Fs = 128; % Hz

% Trim data to contain 4 seconds
ds = Fs*4; % number of samples to select
ini_point = 300; % initial sample of the interval
x = ecg(300:300+ds-1); % trimming
x = normalize(x); % normalize signal using z-scores method

% Confirm if the trimmed signal contains at least 3 QRS complexes
figure;
plot(x)
xlabel('Sample number')
ylabel('Amplitude [uV]')
hold on
% Yes, the plot confirms it contains 6 QRS complexes.

% Add a random noise component with zero mean and 0.4 SD
noise = 0.4*randn(1,ds);
xn = x + noise;

% Plot noised signal on the same original axis
plot(xn)
legend({'Signal X', 'Signal Xn (with noise)'})
title('Original ECG signal and with added noise')
axis tight

% Get SNR of signal xn
snr = signal_to_noise_ratio(xn, noise);
disp("SNR of xn = " + snr + ' dB')
% it returns around 8 dB


%% Exercise b)

% Let us try different error margins:
error_margins = [1e-6 1e-4 1e-3 1e-1];

% Define maximum shifts limit
MAX_SHIFTS = 500;

% Keep the minimum MSE
min_mse = 100; % arbitrary high value
error_min_mse = 0; % the error margin that yielded the minimum MSE

for i=1:length(error_margins) % for each error margin, do:
    fprintf('\n\n####################\nTesting an error margin of %d\n', error_margins(i));
    
    % On the following calls, pass show=true to plot the results.
    
    % Get EMD decomposition of the original signal
    fprintf('\nComputing EMD decomposition of signal X...\n')
    emd_decomposition(x, error_margins(i), MAX_SHIFTS, 'Signal X', false);
    
    % Get EMD decomposition of the noisy signal
    fprintf('\nComputing EMD decomposition of signal Xn...\n')
    imf = emd_decomposition(xn, error_margins(i), MAX_SHIFTS, 'Signal Xn', false);
    
    % Remove noise of the noisy signal
    fprintf('\nRemoving noise from signal Xn...\n')
    [xnf, error] = remove_noise(xn, imf, 'Signal Xn', false);
    
    % For quality accessment, determine the reconstruction/original MSE 
    mse = mean(abs(x-xnf).*abs(x-xnf));
    fprintf('Achieved MSE: %d\n', mse);
    
    % Update minimum MSE
    if mse < min_mse
        min_mse = mse;
        error_min_mse = error_margins(i);
    end
end

fprintf('\nFrom the tested error margins, the one that yielded the minimum MSE was %d (mse = %d)\n', error_min_mse, min_mse);
% We expect the less the error margin is, the less the MSE is going to be.


%% c)

figure(3)

%% d)

noise_2 = 1+0.8*randn(ds,1);
noise_2 = noise_2';
xn2 = x + noise_2;

tic

[imf_xn2,residual_xn2] = emd(xn2,'MAXITERATIONS',500); 

toc


figure(4)
plotimf(xn2,imf_xn2)

%% e)


tic
[imf_100,residual_100] = emd(xn2,'MAXITERATIONS',100);
toc

tic
[imf_1000,residual_1000] = emd(xn2,'MAXITERATIONS',1000); 
toc


figure(5)
plotimf(xn,imf_100)

figure(6)
plotimf(xn,imf_1000)


%% f)

%r(0) = -1
%r(4s) = 2

r = (0:512-1)*(5/512) - 1;
    
xr = x + r;

[xrma] = ma(xr,50);

[imf_xr,residual_xr] = emd(xr); 

errma = x - xrma;
%erremd = x - xremd;

figure(7)

plot(x)
hold on
plot(xr)
hold on
plot(errma)
hold on
%plot(erremd)    
hold on

%% g)

xreemd = eemd(xr, 0, NR, 100);
erreemd = x - xreemd;

figure(8)

plot(x)
hold on
plot(xr)
hold on
plot(errma)
hold on
plot(erreemd)    
hold on
%% h)

xriceemd = ceeemdan(xr, 0, NR, 100);

erreemd = x - xriceemd;

figure(9)

plot(x)
hold on
plot(xr)
hold on
plot(errma)
hold on
plot(erriceemd)    
hold on


%% Auxiliary functions


% Computes the signal to noise ratio, given a signal and its known noise
% conttribution. The result is returned in dB
function ratio = signal_to_noise_ratio(signal, noise)
    % ratio = snr(signal); % dB
    ratio = 20*(log10(rms(signal)/rms(noise))); % dB
    return
end


% EMD decomposition and plots 
function imf = emd_decomposition(signal, error, max_shifts, label, show)

    % Compute IMF
    emd_options = emdoptimset('Stopping', 'Single_T', 'T', error, 'MaxN', max_shifts);
    t0 = tic;
    imf = emd(signal, emd_options);
    toc(t0); % measure computation time
    
    % Plot IMF and residue
    if show == true
        plotimf(signal, imf, label);
    end
    
    return
end


% Reconstructs a signal, removing its noise based on a given IMF computed
% by EMD. Use emd_decomposition first to get the IMF of your signal.
function [reconstructed_signal, error] = remove_noise(signal, imf, label, show)

    % Get IMF size
    [l_imf, len_imf] = size(imf);

    % Allocate memory and initialize to zeros
    reconstructed_signal = zeros(1,len_imf);
    
    % Remove the first IMF
    for i = 1:(l_imf-1)
        reconstructed_signal = reconstructed_signal + imf(l_imf+1-i,:);
    end
    
    % Compute error
    error = signal - reconstructed_signal;
    
    % Plot given signal, reconstructed signal and the error.
    if show == true
        figure;
        subplot(3,1,1);
        plot(signal);
        title (label);
        ylabel ('Amplitude [uV]');
        axis tight
        subplot(3,1,2);
        plot(reconstructed_signal);
        title (strcat(label, ' Reconstructed'));
        ylabel ('Amplitude [uV]');
        axis tight
        subplot(3,1,3);
        plot(error);
        title ('Error');
        ylabel ('Amplitude [uV]');
        axis tight
        sgtitle('Noise reduction')
    end

    return
end




