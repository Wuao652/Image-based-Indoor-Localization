function [angle] = angleDifference_so3(s)
%ANGLEDIFFERENCE Summary of this function goes here
%   Detailed explanation goes here
deltaR = expm(skew(s));
if(abs(trace(deltaR)-3) < 1e-10)
    angle = 0;
else
    angle = acos((trace(deltaR)-1)/2)*360/2/pi;
end
end

