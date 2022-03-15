function [dis] = disDifference(pt1,pt2)
%ANGLEDIFFERENCE Summary of this function goes here
%   Detailed explanation goes here
if(size(pt1,2) ==3)
    pt1 = pt1';
end
if(size(pt2,2) ==3)
    pt2 = pt2';
end
tmp = pt1 - pt2;
dis = sqrt(tmp'*tmp);

end

