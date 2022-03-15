function [output] = huber(x,sigma)
%HUBER Summary of this function goes here
%   Detailed explanation goes here
output = 0;
if(abs(x) <= sigma)
    output = abs(x);
else
    output = sqrt(2*sigma*(abs(x) - 0.5*sigma));
end
end

