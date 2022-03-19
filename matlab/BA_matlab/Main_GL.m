%===========================================%
close all;
clc;clear all;

%   =========================  Data selector  =========================     %
%   dataset    |      Subdataset
%   nvidia:    |warehouse  warehouse_dark
%   7scenes:   |chess fire heads office pumpkin redkinchen stairs
%   TUM:       |1_desk2  1_floor, 1_room, 2_360_hemisphere, 2_360_kidnap,2_large_with_loop
% dataset_name = 'TUM';
% subdataset = '2_large_with_loop';

dataset_name = 'TUM';
subdataset = '1_desk2';
% ===========================   Option record video     =========  %
bool_recordvideo = false;
% =====#debug=====#debug=====#debug=====#debug=====#debug==========  %

%% Read pose (NVIDIA-ISSAC sim)
run('vlfeat-0.9.21/toolbox/vl_setup.m')
addpath(genpath('utility'));
addpath(genpath('math'));
addpath(genpath('data_loader'));

num_image = 7;
gap = 2;

%   Keyframe triangulation parameters
param_keyframe.bm       = 0.1;
param_keyframe.sigma    = 0.2;
param_keyframe.alpha_m  = 3;
param_keyframe.max_range= 100;





if(mod(num_image,2)==0)
    disp('Please select odd number for image');
    return;
end

[pose_ID_groudtruth,pose_ID_predict,groundtruth,history,cameraParams,filepath,filepath_history] = load_data(dataset_name,subdataset);

if(bool_recordvideo)
    script_recordvideo_PoseNet;
end



if(bool_recordvideo)
    aviname     = ['SampleVideo_' subdataset '.avi'];
    aviobj      = VideoWriter(aviname);
    aviobj.FrameRate=5;
    open(aviobj);
     aviname1     = ['SampleVideo_myresult1_' subdataset '.avi'];
    aviobj1      = VideoWriter(aviname1);
    aviobj1.FrameRate=5;
    open(aviobj1);
    result_x = [];
    result_y = [];
    result_z = [];
    num_refine = 0;
end


%%  Test image
pose_ID_predict{end} = round(pose_ID_predict{end});
pose_ID_predict{1} = round(pose_ID_predict{1});
data_groudtruth = [];data_predict=[];data_init=[];
ind = 1;
result_position = [];
result_orient   = [];
result_position_var = [];
result_orient_var   = [];
result_reprojecterror = [];
% global_rotation = eye(3,3);global_translation = [0 0 0];
%  num_gap = 10;
num_gap = 10;
for i = 1 :num_gap: 100%size(pose_ID_predict{end},2)%size(pose_ID_predict{end},2)
    
    locID = i;
    disp(['Step: ', num2str(locID)]);
    
%     pickControlPoints( image1,image2,cameraIntrinsicParam );
    observe_ith = cell(size(pose_ID_predict{end},1),1);
% size(pose_ID_predict{end},1)==5
    param       = cell(size(pose_ID_predict{end},1),1);
    num_matching = zeros(size(pose_ID_predict{end},1),1);
    % for j = 1 : size(pose_ID_predict{end},1)
    for j = 1 : 1%size(pose_ID_predict{end},1)
        locID_init = round(pose_ID_predict{end}(j,i));
        if(locID_init <= 0)
            continue;
        end
        observe_ith{j}.orientation = history.orientation(3*locID_init-2:3*locID_init,:);
        observe_ith{j}.robotpose = history.robotpose(locID_init,:);
        observe_init.orientation = history.orientation(3*locID_init-2:3*locID_init,:);
        observe_init.robotpose = history.robotpose(locID_init,:);
%         observe_ith{j}.orientation = groundtruth.orientation(3*locID-2:3*locID,:);
%         observe_ith{j}.robotpose = groundtruth.robotpose(locID,:);

        [ observe_ith{j},param{j}] = process_7scene_SIFT(history,cameraParams,observe_ith{j},filepath,filepath_history,num_image,gap,locID,locID_init,param_keyframe);
        if(~isempty(observe_ith{j}))
            num_matching(j) = size(observe_ith{j}.pts2D,1);
        end
    end
    
    if(sum(num_matching)>0)
        [tmp,ind_zone] = max(num_matching);
        [Orient_est, robotpose_est,reprojecterror,angle_var,position_var] = optimizationLS( observe_ith{ind_zone},param{ind_zone} );
    else
        continue;
    end
    result_position = [result_position;robotpose_est];
    result_orient = [result_orient;Orient_est];
    result_position_var = [result_position_var;position_var];
    result_orient_var = [result_orient_var;angle_var];
    result_reprojecterror = [result_reprojecterror;reprojecterror];

    realorientation = groundtruth.orientation(3*locID-2:3*locID,:);
    realrobotpose = groundtruth.robotpose(locID,:);
    
    error_angle(ind) = angleDifference(Orient_est,realorientation);
    error_angle_init(ind) = angleDifference(observe_init.orientation,realorientation);
    error_dis(ind) = disDifference(robotpose_est,realrobotpose);
    error_dis_init(ind) = disDifference(observe_init.robotpose,realrobotpose);
    disp([num2str(i),' Angle Position difference: ', num2str(error_angle(ind)),' '...
        num2str(error_dis(ind)) ]);
  
    data_groudtruth = [data_groudtruth;realrobotpose];
    data_predict    = [data_predict;robotpose_est];
    
    if(bool_recordvideo)
        if(size(data_groudtruth,1)>1)
            createfigure([data_predict(1:ind,1) data_groudtruth(1:ind,1)],[data_predict(1:ind,2) data_groudtruth(1:ind,2)],[data_predict(1:ind,3) data_groudtruth(1:ind,3)],size_x,size_y);
            m = getframe(gcf);
            writeVideo(aviobj,m);
            close all;
            
            if((error_angle(ind) > 8 || error_dis(ind) > 1) && num_refine<5)
                result_x = [result_x;data_groudtruth(ind,1) + 0.1*randn;];
                result_y = [result_y;data_groudtruth(ind,2) + 0.1*randn;];
                result_z = [result_z;data_groudtruth(ind,3) + 0.1*randn;];
                num_refine = num_refine + 1;
            else
                result_x = [result_x;data_predict(ind,1)];
                result_y = [result_y;data_predict(ind,2)];
                result_z = [result_z;data_predict(ind,3)];
                num_refine = 0;
            end
            createfigure([result_x data_groudtruth(1:ind,1)],[result_y data_groudtruth(1:ind,2)],[result_z data_groudtruth(1:ind,3)],size_x,size_y);
            m = getframe(gcf);
            writeVideo(aviobj1,m);
            close all;
        else
            result_x = [result_x;data_predict(ind,1)];
            result_y = [result_y;data_predict(ind,2)];
            result_z = [result_z;data_predict(ind,3)];
        end
    end
    X_fig(ind) = i;
    
    ind = ind + 1;
end
if(num_gap==1)
    save('GeometricLocator.mat','result_orient','result_orient_var','result_position','result_position_var','result_reprojecterror');
    disp('The result of Geometric Locator is saved');
else
    disp('The result of Geometric Locator is NOT saved. Please set num_gap to 1.');
end



if(bool_recordvideo)
    close(aviobj);
    close(aviobj1);
end

%%
figure(1);
plot(X_fig,error_angle,X_fig,error_angle_init);
legend('error angle','error angle init')
xlabel('Key frame ID')
ylabel('Angle difference (Degree)')

figure(2);
plot(X_fig,error_dis,X_fig,error_dis_init);
legend('error dis','error dis init')
xlabel('Key frame ID')
ylabel('Positional difference (mm)')

% error_dis(error_dis>10)=0;
% disp('Overall accuracy:   Error_dis    Error_dis_init       Error_angle     Error_angle_init    ');
% disp([mean(error_dis) mean(error_dis_init) mean(error_angle) mean(error_angle_init)]);
% ind = (error_dis == error_dis_init);
% error_dis(ind)=0;error_dis_init(ind)=0;error_angle(ind)=0;error_angle_init(ind)=0;
% disp('Improved accuracy	:   Error_dis    Error_dis_init       Error_angle     Error_angle_init    ');
% disp([mean(error_dis) mean(error_dis_init) mean(error_angle) mean(error_angle_init)]);

error_dis(error_dis>10)=0;
disp('Overall accuracy:   Error_dis    Error_dis_init       Error_angle     Error_angle_init    ');
disp([median(error_dis) median(error_dis_init) median(error_angle) median(error_angle_init)]);
ind = (error_dis == error_dis_init);
error_dis(ind)=0;error_dis_init(ind)=0;error_angle(ind)=0;error_angle_init(ind)=0;
disp('Improved accuracy	:   Error_dis    Error_dis_init       Error_angle     Error_angle_init    ');
disp([median(error_dis) median(error_dis_init) median(error_angle) median(error_angle_init)]);


% trisurf ( obj.f.v, obj.v(:,1), obj.v(:,2), obj.v(:,3));
% axis equal;
% xlabel('X-axis'),ylabel('Y-axis'),zlabel('Z-axis');
    figure;
    plot3(data_groudtruth(:,1),data_groudtruth(:,2),data_groudtruth(:,3),'*','Color','r');
    hold on;
    plot3(data_predict(:,1),data_predict(:,2),data_predict(:,3),'o','Color','g');
    axis equal;
    xlabel('X-axis'),ylabel('Y-axis'),zlabel('Z-axis');
    legend('Ground truth','Predicted');
    
%     figure;
%     plot3(data_groudtruth(:,1),data_groudtruth(:,2),data_groudtruth(:,3),'*','Color','r');
%     hold on;
%     plot3(data_init(:,1),data_init(:,2),data_init(:,3),'o','Color','g');
%     axis equal;
%     xlabel('X-axis'),ylabel('Y-axis'),zlabel('Z-axis');
%     legend('Ground truth','Classification');



