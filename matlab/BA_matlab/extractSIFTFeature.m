function [featuresPrev,pointsPrev] = extractSIFTFeature(I)
%EXTRACTSIFTFEATURE Summary of this function goes here
%   Detailed explanation goes here
I = single(I);
[f, d] = vl_sift(I,'EdgeThresh',20,'Octaves',4,'Levels',6) ;
featuresPrev = d';
pointsPrev = f(1:2,:)';
end

