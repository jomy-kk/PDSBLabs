function hht_imf(imf,Ts)
% Plot the HHT.
% hht_emdo(imf,Ts)
% :: Syntax
%    The cell array imf is the input IMF & residue and Ts is the sampling period.
%    Example on use: [x,Fs] = wavread('Hum.wav');
%                    options = emdoptimset('stop','Nsifts','N',500,'imfs',a);
%                    [IMF,imf] = emdo(x,options);
%                    hht_emd0(imf,1/Fs);
M = length(imf);
N = length(imf{1});
for k = 1:M
   b(k) = sum(imf{k}.*imf{k});
   th   = angle(hilbert(imf{k}));
   d{k} = diff(th)/Ts/(2*pi);
end
[u,v] = sort(-b);
b     = 1-b/max(b);
q = linspace(0.1,1,M);
r = linspace(0.1,1,M);
% Set time-frequency plots.
c = linspace(0,(N-2)*Ts,N-1);
figure;
for k = v(1:M)
   %plot(c,d{k},'k.','Color',b([k k k]),'MarkerSize',5);
   plot(c,d{k},'k.','Color',[b(k) q(k) r(k)],'MarkerSize',5);
   set(gca,'FontSize',8,'XLim',[0 c(end)],'YLim',[0 1/2/Ts]); xlabel('Time'), ylabel('HHT Frequency');
   hold on;
end

% Set IMF plots.
c = linspace(0,(N-1)*Ts,N);
for k1 = 0:4:M-1
   figure;
   set(gca,'FontSize',8,'XLim',[0 c(end)]);   
   for k2 = 1:min(4,M-k1), subplot(4,1,k2), plot(c,imf{k1+k2});
        ylabel(['Imf' num2str(k1+k2)]);
   end
   xlabel('Time');
end
