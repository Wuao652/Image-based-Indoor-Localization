function [ F,W ] = CalculateF_LS( observe,param,x,Orient,bool_outlier )
%===========================================%
%   Deformable SLAM system
%   Method:     1.Base mixing based deformable least square.
%               2.Traditional least square
%   Input:      observe:        Robot to features observations
%               param:          Parameters of observation and robot
%               x:              X state
%               Orient:         Orientation of robot
%               Huber loss:     
%   Output:     F:              Output error function
%===========================================%
bool_outlier = false;
num_F = 2*param.num_features;
F = zeros(num_F,1);
w = zeros(num_F,1);
K = param.K;

%   Function I: Robot to feature observation
robotpose = x(4:6);
R = Orient;
ind = 1;
for j = 1 : param.num_features
    feature   = observe.pts3D(j,:)';
    feature_cam = R*(feature - robotpose);
    %ind = (i-1)*2*param.num_features + (j-1)*2 + 1;
    v      = K(1,1)*feature_cam(1)/feature_cam(3) + K(1,3);
    u      = K(2,2)*feature_cam(2)/feature_cam(3) + K(2,3);
    F(ind:ind+1) = [u;v] - observe.pts2D(j,:)';
    
    %   Robustify
%     delta = 5;
%     if(abs(F(ind))>delta || abs(F(ind+1))>delta)
%         if(abs(F(ind))>delta)
%             F(ind)   = sqrt(2*delta*(abs(F(ind)  )-0.5*delta));
%         end
%         if(abs(F(ind+1))>5)
%             F(ind+1) = sqrt(2*delta*(abs(F(ind+1))-0.5*delta));
%         end
%     end
    w(ind)   = (huber(F(ind),1)/F(ind))^2;
    w(ind+1) = (huber(F(ind+1),1)/F(ind+1))^2;
%     if(abs(F(ind)) > 1)
%         F(ind) = log10(abs(F(ind)));
%     else
%         F(ind) = 0;
%     end
%     if(abs(F(ind+1)) > 1)
%         F(ind+1) = log10(abs(F(ind+1)));
%     else
%         F(ind+1) = 0;
%     end
    ind = ind + 2;
end

W = diag(w);
% sigma = std(F);
if(bool_outlier == true)
    ind = 1;
    delta = 20;
    for j = 1 : param.num_features
        if(abs(F(ind))>delta || abs(F(ind+1))>delta)
            F(ind:ind+1) = 0;
        end
        ind = ind + 2;
    end
end

end

