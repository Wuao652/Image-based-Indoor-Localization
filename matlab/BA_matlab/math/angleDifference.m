function [angle] = angleDifference(R1,R2)
%ANGLEDIFFERENCE Summary of this function goes here
%   Detailed explanation goes here
if(abs(trace(R1'*R2)-3) < 1e-10)
    angle = 0;
else
    angle = acos((trace(R1'*R2)-1)/2)*360/2/pi;
end
end

