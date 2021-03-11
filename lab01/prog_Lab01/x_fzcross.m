% x_fzcross.m
% commented find zero crossings
% Neil Jerome 13 July 2020
%% set up arbitrary signal (sum of 2 sines for demo)
T = 0.01;
x = 0:T:4;
sin1 = 10*sin(10*x); sin2 = 42*sin(6*x);
signal = sin1+sin2;
% plot to show signal
figure; 
plot(x, signal, 'b.-');
hold on;
plot(x, zeros(1,length(x)), 'k:');
%% find next point after zero crossing
sigPos = logical(signal>0); % find all positive points
cross = sigPos - circshift(sigPos,1); % find changeover points 
% add to plot
plot(x(logical(cross)), signal(logical(cross)), 'ko');
%% assume straight line approximation between adjacent points
crossInd = find(cross); % x indices of cross points
nCross = length(crossInd); % number of cross points found
x0 = NaN(1,nCross); % vector of x-values for cross points
for aa = 1:nCross
    thisCross = crossInd(aa);
    
    % interpolate to get x coordinate of approx 0
    x1 = thisCross-1; % before-crossing x-value
    x2 = thisCross; % after-crossing x-value
    y1 = signal(thisCross-1); % before-crossing y-value
    y2 = signal(thisCross); % after-crossing y-value
    
    ratio = (0-y1) / (y2-y1); % interpolate between to find 0
    x0(aa) = x(x1) + (ratio*(x(x2)-x(x1))); % estimate of x-value
    
end
    
% add to plot
plot(x0, zeros(1,nCross), 'ro');