% Lab 6
%% a)
load('ecg.mat')
Fs = 128;
ds = Fs*4;
ini_point = 300;
x = ecg(300:300+ds-1);
noise = 0.4*randn(ds,1);
noise = noise';
xn = x + noise;

r = snr(x,noise);

%% b)
tic
[imf,residual] = emd(xn,'MAXITERATIONS',500); 
toc     

figure(2)
plotimf(xn,imf)

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




