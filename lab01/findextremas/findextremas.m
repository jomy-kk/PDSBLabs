function [st, maximas, minimas, ed]=findextremas(h)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% findextremas - finds maximas and minimas (i. e. peaks or extremas) of
% given signal x
% INPUT:
%
% - h: signal in a one dimensional array
% 
%
% OUTPUT:
% -st: x,y coordinates of Start point, 
% -maximas: x,y coordinates of maxima points/ peaks,  
% -minimas: x,y coordinates of minima points/ peaks,  
% -ed: x,y coordinates of Start point,
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


lh=length(h);
df=[];maximas=[];minimas=[];
for j=2:lh-1
   
    df(j)=h(j)-h(j-1);
    df(j+1)=h(j+1)-h(j);
    
    if df(j)==0 || ( df(j)>=0 && df(j+1)<0 )
    
       maximas=[maximas; j,h(j)];
    elseif df(j)==0 || ( df(j)<=0 && df(j+1)>0 )
        minimas=[minimas; j,h(j)];
    end
end
%  maximas=[1, h(1);maximas;lh, h(lh) ];
%  minimas=[1, h(1);minimas;lh, h(lh) ];
st=[1, h(1)];
ed=[lh, h(lh)];
pks=[st; maximas; minimas; ed];
end