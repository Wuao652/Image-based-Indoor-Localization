function [ind_neighbor] = selectimage(orientation,robotpose,ind_1,bm,sigma,alpha_m,num_img,max_range)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Function: Select the images for triangulation. Following section 4.1 in Mobile3Dcon
%   This searching is on both directions, left and right
%   Method: score=wb*wv 
%           wb=exp(-(dist_position-bm)^2/sigma^2);
%           wv=max(m/(t, t'),1),
%   Go two direction of image, find one, set it as seed, go for next
%   Input: 
%   scale:          Scale of the pose
%   orientation:    Mat 3 * 3
%   robotpose:      [x y z]
%   ind_1:          index of the left (first) image
%   bm:             length of the baseline
%   sigma:          parameter of baseline
%   alpha_m:        parameter of orientation, maximum angle (degree)
%   num_img:        Number of images to be selected
%   max_range:      Maximum range to choose the optimal keyframe
%   Returns:
%   ind_neighbor:   The index of the selected images.
%   Author:   Jingwei Song.   15/07/2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ind_neighbor_l = [];
ind_neighbor_r = [];
seed_l = ind_1;
seed_r = ind_1;
while(size(ind_neighbor_l,1)+size(ind_neighbor_r,1) < num_img)
    %   Right hand searching
    ind = selectImage_onedirection(orientation,robotpose,seed_l,bm,sigma,alpha_m,max_range,1);
    if(ind > 0)
        ind_neighbor_l = [ind_neighbor_l;ind];
        seed_l = ind;
    end
    if(size(ind_neighbor_l,1)+size(ind_neighbor_r,1) == num_img)
        break;
    end
    %   Left hand saerching
    ind = selectImage_onedirection(orientation,robotpose,seed_r,bm,sigma,alpha_m,max_range,-1);
    if(ind > 0)
        ind_neighbor_r = [ind_neighbor_r;ind];
        seed_r = ind;
    end
    if(size(ind_neighbor_l,1)+size(ind_neighbor_r,1) == num_img)
        break;
    end
end
ind_neighbor = [flip(ind_neighbor_r);ind_neighbor_l];
end

function [ind_2] = selectImage_onedirection(orientation,robotpose,ind_1,bm,sigma,alpha_m,max_range,sign)

T_l = zeros(4,4);   %   Camera pose left  in SE(3). Body to world
T_r = zeros(4,4);   %   Camera pose right in SE(3). Body to world
T_l(1:3,4)   = robotpose(ind_1,1:3)';
T_l(1:3,1:3) = orientation(3*ind_1-2:3*ind_1,:);

score = zeros(max_range,1);
for i = 1:max_range
    if(ind_1+i*sign < 1 || ind_1+i*sign > size(robotpose,1))
        break;
    end
    T_r(1:3,4)   = robotpose(ind_1+i*sign,1:3)';
    T_r(1:3,1:3) = orientation(3*(ind_1+i*sign)-2:3*(ind_1+i*sign),:);
    dist_position = norm(T_l(1:3,4)-T_r(1:3,4));
    dist_angle    = angleDifference(T_l(1:3,1:3),T_r(1:3,1:3));
    wb = exp(-(dist_position-bm)^2/sigma^2);
    wv = min(alpha_m/dist_angle,1);
    score(i) = wb*wv;
end

[score_descend,ind]=sort(score,'descend');
ind_2 = ind_1 + ind(1)*sign;

if(score_descend(1)==0)
    ind_2 = 0;
end

end

