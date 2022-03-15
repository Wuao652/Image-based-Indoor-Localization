function [ x] = getskew( skew_matrix )
%GETSKEW Summary of this function goes here
%   Detailed explanation goes here
% skew_matrix=[0 -x(3) x(2) ;...
%     x(3) 0 -x(1);...
%     -x(2) x(1) 0 ];

x(1) = -skew_matrix(2,3);
x(2) = skew_matrix(1,3);
x(3) = -skew_matrix(1,2);
end

