function plotimfd(x,z,st,sp,label)
figure; %a =length(z(:,1));
b = sp-st+1;
subplot(b+1,1,1); plot(x);
ylabel (['x']);
for i = 2:b+1
    subplot(b+1,1,i); plot(z(st+i-2,:));
    ylabel (['i.' num2str(st+i-2)]);
end
sgtitle(strcat("IMF ", st ," of ", label))
return
