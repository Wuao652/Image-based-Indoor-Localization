function y = skew( x )
[m,n] = size(x);
if(m+n ~= 4)
    error('dimension wrong');
end
y=[0 -x(3) x(2);...
    x(3) 0 -x(1);...
    -x(2) x(1) 0];


end

