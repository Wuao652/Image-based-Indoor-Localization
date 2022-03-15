function [tracks1,tracks2] = create_Tracks(vSet,num_image)
%CREATE_TRACKS Summary of this function goes here
%   Detailed explanation goes here
%   Tracks1: Tracks including the last image
%   Tracks2: Tracks excluding the last image

%   Generate tracks 1
ind = 1:1:num_image;
tracks = findTracks(vSet,ind);
tmp = [];
k = 1;
while(k<=size(tracks,2))
    if(size(tracks(1,k).ViewIds,2) >2 && tracks(1,k).ViewIds(1,1)==1)
        tmp = [tmp tracks(1,k)];
    end
    k = k + 1;
end
tracks1 = tmp;


%   Generate tracks 2
tracks2 = tracks1;
for i = 1 : size(tracks2,2)
    tracks2(1,i).ViewIds =tracks2(1,i).ViewIds(2:end);
    tracks2(1,i).Points = tracks2(1,i).Points(2:end,:);
end
end

