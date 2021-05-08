% Lab 6

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
disp("SNR of xn = " + signal_to_noise_ratio(xn, noise) + ' dB')
% it returns around 8 dB


%% Exercise b)

% Let us try different error margins:
error_margins = [1e-6 1e-4 1e-3 1e-1];

% Define maximum shifts limit
MAX_SHIFTS = 500;

% Run tests
[xnf, imf] = emd_and_remove_noise(x, 'Signal X', xn, 'Signal Xn', MAX_SHIFTS, error_margins, 1, true);


%% Exercise c)

% Professor implementation
options = emdoptimset('stop','Nsifts','N',MAX_SHIFTS);
imf = emd(xn, options);
imf = num2cell(imf, 2); % convert matrix to cell
hht_imf(imf, 1/Fs);

% Our implementation
my_pHHT(xn, 1/Fs);


%% Exercise d)

% Add a random noise component with mean of 1 and SD of 0.8
noise_2 = 1 + 0.8*randn(1, ds);
xn2 = x + noise_2;

% Let us try different error margins:
error_margins = [1e-6 1e-4 1e-3 1e-1];

% Define maximum shifts limit
MAX_SHIFTS = 500;

% Run tests
[xnf2, imf2] = emd_and_remove_noise(x, 'Signal X', xn2, 'Signal Xn2', MAX_SHIFTS, error_margins, 1, true);


%% Exercise e)

% Let us try different error margins:
error_margins = [0.005 0.05 0.001];

% Run tests for 100 shifts maximum
MAX_SHIFTS = 100;
[xnf2, imf2] = emd_and_remove_noise(x, 'Signal X', xn2, 'Signal Xn2', MAX_SHIFTS, error_margins, 1, true);

% Run tests for 1000 shifts maximum
MAX_SHIFTS = 1000;
[xnf2, imf2] = emd_and_remove_noise(x, 'Signal X', xn2, 'Signal Xn2', MAX_SHIFTS, error_margins, 1, true);


%% Exercise f)

% Add a linear ramp noise component from r(0)=-1 to r(4s)=2
linear_ramp_noise = (0:512-1)*(5/512) - 1;
xr = x + linear_ramp_noise;

%Try to remove the noise with a MA filter of order 50
trend = ma(xr,100);
xrma = xr - trend;

% Plot results
figure;
plot(xr);
hold on
plot(trend);
hold on
plot (xrma);
hold on
legend({'Signal Xr', 'MA trend', 'Signal Xrma'})
title('Removing noise with MA')
axis tight

% Try to remove the noise using EMD, by removing the residue (-1 on the 7th argument)
[xremd, ~] = emd_and_remove_noise(x, '', xr, '', 100, [0.0001, ], -1, false);

% Plot results
figure;
plot(xr);
hold on
plot (xremd);
hold on
legend({'Signal Xr', 'Signal Xremd'})
title('Removing noise with EDM')
axis tight

% Determine errors
errma = x - xrma;
erremd = x - xremd;

% Plot what is asked
figure(7);
plot(x);
hold on
plot(xr);
hold on
plot(errma);
hold on
plot(erremd);
hold on
legend({'Signal X', 'Signal Xr', 'MA error', 'EMD error'})
title('Comparison of noise removal errors between MA and EMD methods')
axis tight

% Determine MSEs
fprintf('MSE of MA: %d\n', mean(abs(x-xrma).*abs(x-xrma)))
fprintf('MSE of EMD: %d\n', mean(abs(x-xremd).*abs(x-xremd)))


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
function [reconstructed_signal, error] = remove_noise(signal, imf, remove, label, show)

    % Get IMF size
    [l_imf, len_imf] = size(imf);

    % Allocate memory and initialize to zeros
    reconstructed_signal = zeros(1,len_imf);
    
    if remove == 1
        disp("Removing the first IMF...")
        % Remove the first IMF
        for i = 1:(l_imf-1)
            reconstructed_signal = reconstructed_signal + imf(l_imf+1-i,:);
        end
    end
    
    if remove == -1
        disp("Removing the residue...")
        % Remove the residue
        for i = 1:(l_imf-1)
            reconstructed_signal = reconstructed_signal + imf(i,:);
        end
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


% Uses emd_decomposition and remove_noise to perfom multiple tests on
% diffent EMD-based reconstructions of variable margin errors.
% Pass show=true to plot the results.
% It returns the best reconstructed signal that yielded a less MSE with the
% original signal. It also returns the IMFs of the respective EMD.
function [best_reconstructed_signal, best_imf] = emd_and_remove_noise(original_signal, original_label, noisy_signal, noisy_label, max_shifts, error_margins, remove, show)
    
    % Keep the minimum MSE
    min_mse = 100; % arbitrary intial high value
    
    for i=1:length(error_margins) % for each error margin, do:
        fprintf('\n\n####################\nTesting an error margin of %d\n', error_margins(i));

        % Get EMD decomposition of the original signal
        fprintf('\nComputing EMD decomposition of %s ...\n', original_label)
        emd_decomposition(original_signal, error_margins(i), max_shifts, original_label, show);

        % Get EMD decomposition of the noisy signal
        fprintf('\nComputing EMD decomposition of %s ...\n', noisy_label)
        imf = emd_decomposition(noisy_signal, error_margins(i), max_shifts, noisy_label, show);

        % Remove noise of the noisy signal
        fprintf('\nRemoving noise from %s ...\n', noisy_label)
        [reconstructed_signal, error] = remove_noise(noisy_signal, imf, remove, noisy_label, show);

        % For quality accessment, determine the reconstruction/original MSE 
        mse = mean(abs(original_signal-reconstructed_signal).*abs(original_signal-reconstructed_signal));
        fprintf('Achieved MSE: %d\n', mse);

        % Update minimum MSE
        if mse < min_mse
            min_mse = mse;
            error_min_mse = error_margins(i);
            best_reconstructed_signal = reconstructed_signal;
            best_imf = imf;
        end
    end

    fprintf('\nFrom the tested error margins, the one that yielded the minimum MSE was %d (mse = %d).\n', error_min_mse, min_mse);
    % We expect the less the error margin is, the less the MSE is going to be.

    fprintf('Returning that as the best reconstructed signal.\n');

    % Compute SNR
    disp("SNR of the best reconstructed signal = " + snr(reconstructed_signal) + ' dB')

    % Delete auxiliary variables
    clear('error_min_mse')
    clear('best_xn2f')
    clear('best_imf2')
    clear('min_mse')
    
    return
    
end


% Our implementation of HHT
function my_pHHT(x, Ts)
% Given the pre-computed IMFs of a signal (including the residue) and Ts
% the sampling period, it plots the HHT.

% Compute IMF
imfA = emd_decomposition(x, 1e-6, 500, 'Signal X', false); %computation of imfAs
imf = num2cell(imfA, 2); % convert matrix to cell, respective to the 2nd axis

% Some constants
M = length(imf);
N = length(x);

% Compute Hilbert-Huang transform 
for i = 1:M
    z{i} = hilbert(imf{i}); % get the complex
    b(i) = sum(imf{i}.*imf{i}); % get total energy of each IMF
    th   = angle(z{i}); % get the phase of the complex
    d{i} = diff (th)/(Ts*2*pi); % get phase derivatives
end

% Get relative energy of each IMF
b = b/max(b); 

% Plots time vs frequency vs amplitude (spectogram)
c = linspace(0, (N-2)*Ts, N-1);
figure1 = figure();
figure2 = figure();

for k = 1:M
    power_index = b(k);
    if power_index<0.5
        colors = [0 2*power_index 1-2*power_index];
    else 
        colors = [2*(power_index-0.5) 1-2*(power_index-0.5) 0];
    end
    figure(figure1)
    plot(c,d{k},'*','color',colors,'MarkerSize',2);
    set(gca,'FontSize',8,'XLim',[0 c(end)],'YLim',[0 1/(2*Ts)]); xlabel('Time'), ylabel('Frequency');
    hold on 
    figure(figure2)
    plot(c,d{k},'*','color',colors,'MarkerSize',3);
    set(gca,'FontSize',8,'XLim',[0 c(end)],'YLim',[0 1/(2*Ts)]); xlabel('Time'), ylabel('Frequency');
    hold on
    figure(figure2)
    plot(c,d{k},'color',colors,'Linewidth', 1e-6);
    set(gca,'FontSize',8,'XLim',[0 c(end)],'YLim',[0 1/(2*Ts)]); xlabel('Time'), ylabel('Frequency');
    hold on
end



% try to use this function given. It's giving an error.
%[A,f,tt] = hhspectrum(x, c);
%figure
%plot(tt, f, '*','color', A, 'MarkerSize',2);


% Plot IMFs
figure
sgtitle('EMD Decomposition')
c = linspace(0, (N-1)*Ts, N); % horizontal axis
for k1 = 0:M
   if k1 == 0 % first one
       title ('Original Signal');
       subplot(M+1,1,k1+1);
       plot(c,x);
       xlabel('Time [s]');
       ylabel ('Amplitude [uV]');
       set(gca,'FontSize',8,'XLim',[0 c(end)]);
   else % following ones
       subplot(M+1,1,k1+1);
       plot(c,imf{k1});
       xlabel('Time [s]');
       ylabel ('Amplitude [uV]');
       if k1 ~= M % not the last one
           title (['IMF ' num2str(k1)]);
       else % last one is the residue
           title ('Residue');
       end
       set(gca,'FontSize',8,'XLim',[0 c(end)]);
   end
end

end

