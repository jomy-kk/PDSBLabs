function [ y ] = ma( x,N ) % Moving Average order N
buf = zeros(1,N); ia=1; sum = 0;
for i=1:length(x)
	sum = sum - buf(ia) + x(i);	
	buf(ia) = x(i);
    	ia = mod(ia,N)+1;
	y(i) = sum/N;
end
end
