function [ J ] = JacobianF_LS( observe,param,x,Orient )
%===========================================%
%   Deformable SLAM system
%   Method:     1.Base mixing based deformable least square.
%               2.Traditional least square
%   Input:      observe:        Robot to features observations
%               param:          Parameters of observation and robot
%               x:              X state
%               Orient:         Orientation of robot
%   Output:     F:              Output error function
%===========================================%
num_F = 2*param.num_features;
J = zeros(num_F,6);
K = param.K;


%   Function I: Robot to feature observation
ind = 1;
robotpose = x(4:6);
R = Orient;
for j = 1 : param.num_features
    feature   = observe.pts3D(j,:)';
    feature_cam = R*(feature - robotpose);
    v      = K(1,1)*feature_cam(1)/feature_cam(3) + K(1,3);
    u      = K(2,2)*feature_cam(2)/feature_cam(3) + K(2,3);
    
    %   Temp: partial(PI)/partial [S=R(f-p)=feature_cam]
    PI_feature_cam = [0 K(2,2)/feature_cam(3) -K(2,2)*feature_cam(2)/feature_cam(3)^2;
        K(1,1)/feature_cam(3) 0 -K(1,1)*feature_cam(1)/feature_cam(3)^2];
    
    %   partial(F)/partial(pose_orient)
    J(ind:ind+1,1:3) = PI_feature_cam * (-R*skew(feature - robotpose));
    
    %   partial(F)/partial(poseX,poseY,poseZ)
    J(ind:ind+1,4:6) = PI_feature_cam * (-R);
    
    ind = ind + 2;
end

end
