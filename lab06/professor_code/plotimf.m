function plotimf(x,z)
figure; a = length(z(:,1));
subplot(a+1,1,1); plot(x);
ylabel (['x']); axis tight;
for i = 2:a+1
    subplot(a+1,1,i); plot(z(i-1,:));
    ylabel (['i.' num2str(i-1)]); axis tight
end
return
