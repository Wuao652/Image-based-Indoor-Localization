function [ skew_matrix] = getskew( x )
%GETSKEW Summary of this function goes here
%   Detailed explanation goes here

[m,n] = size(x);
if(m+n ~= 4)
    error('dimension wrong');
end

skew_matrix=[0 -x(3) x(2) ;...
    x(3) 0 -x(1);...
    -x(2) x(1) 0 ];


end

