switch dataset_name
    case 'TUM'
        driverpath = 'E:\dataset\FXPAL\dataset_TUM\rgbd_dataset_freiburg';
        switch subdataset
            case '1_desk2'
                disp('[PoseNet record] TUM 1_desk2 data set');
                filename = [driverpath  subdataset '\posenet_training_output\cdf\2020_05_02_13_10_train_data_training_allseq_b256_e360_lr0001_b250_output_1.h5py'];
                posenet_x = h5read(filename,'/posenet_x_predicted');posenet_x=posenet_x';
                posenet_y = h5read(filename,'/posenet_y_predicted');posenet_y=posenet_y';
                posenet_z = h5read(filename,'/posenet_z_predicted');posenet_z=posenet_z';
            case '1_floor'
                disp('[PoseNet record] TUM 1_floor data set');
                filename = [driverpath  subdataset '\posenet_training_output\cdf\2020_05_02_17_43_train_data_training_allseq_b256_e360_lr0001_b250_output_1.h5py'];
                posenet_x = h5read(filename,'/posenet_x_predicted');posenet_x=posenet_x';
                posenet_y = h5read(filename,'/posenet_y_predicted');posenet_y=posenet_y';
                posenet_z = h5read(filename,'/posenet_z_predicted');posenet_z=posenet_z';
            case '1_room'
                disp('[PoseNet record] TUM 1_floor data set');
                filename = [driverpath  subdataset '\posenet_training_output\cdf\2020_05_02_17_43_train_data_training_allseq_b256_e360_lr0001_b250_output_1.h5py'];
                posenet_x = h5read(filename,'/posenet_x_predicted');posenet_x=posenet_x';
                posenet_y = h5read(filename,'/posenet_y_predicted');posenet_y=posenet_y';
                posenet_z = h5read(filename,'/posenet_z_predicted');posenet_z=posenet_z';
            case '2_360_hemisphere'
                disp('[PoseNet record] TUM 1_floor data set');
                filename = [driverpath  subdataset '\posenet_training_output\cdf\2020_05_03_10_14_train_data_training_allseq_b256_e360_lr0001_b250_output_1.h5py'];
                posenet_x = h5read(filename,'/posenet_x_predicted');posenet_x=posenet_x';
                posenet_y = h5read(filename,'/posenet_y_predicted');posenet_y=posenet_y';
                posenet_z = h5read(filename,'/posenet_z_predicted');posenet_z=posenet_z';
            case '2_360_kidnap'
                disp('[PoseNet record] TUM 1_floor data set');
                filename = [driverpath  subdataset '\posenet_training_output\cdf\2020_05_02_17_47_train_data_training_allseq_b256_e360_lr0001_b250_output_1.h5py'];
                posenet_x = h5read(filename,'/posenet_x_predicted');posenet_x=posenet_x';
                posenet_y = h5read(filename,'/posenet_y_predicted');posenet_y=posenet_y';
                posenet_z = h5read(filename,'/posenet_z_predicted');posenet_z=posenet_z';
            case '2_large_with_loop'
                disp('[PoseNet record] TUM 1_floor data set');
                filename = [driverpath  subdataset '\posenet_training_output\cdf\2020_05_02_17_51_train_data_training_allseq_b256_e360_lr0001_b250_output_1.h5py'];
                posenet_x = h5read(filename,'/posenet_x_predicted');posenet_x=posenet_x';
                posenet_y = h5read(filename,'/posenet_y_predicted');posenet_y=posenet_y';
                posenet_z = h5read(filename,'/posenet_z_predicted');posenet_z=posenet_z';
            otherwise
                disp('other value')
        end
    case 'nvidia'
        driverpath = 'E:\dataset\NVIDIA\indoor_dataset\';
        switch subdataset
            case 'warehouse'
                disp('[PoseNet record] TUM 1_desk2 data set');
                filename = [driverpath  subdataset '\posenet_training_output\cdf\2021_11_08_20_33_train_data_training_allseq_b256_e360_lr0001_b250_output_1.h5py'];
                posenet_x = h5read(filename,'/posenet_x_predicted');posenet_x=posenet_x';
                posenet_y = h5read(filename,'/posenet_y_predicted');posenet_y=posenet_y';
                posenet_z = h5read(filename,'/posenet_z_predicted');posenet_z=posenet_z';
            case 'warehouse_dark'
                disp('[PoseNet record] TUM 1_desk2 data set');
                filename = [driverpath  subdataset '\posenet_training_output\cdf\2021_11_11_15_35_train_data_training_allseq_b256_e360_lr0001_b250_output_1.h5py'];
                posenet_x = h5read(filename,'/posenet_x_predicted');posenet_x=posenet_x';
                posenet_y = h5read(filename,'/posenet_y_predicted');posenet_y=posenet_y';
                posenet_z = h5read(filename,'/posenet_z_predicted');posenet_z=posenet_z';
            otherwise
                disp('other value')
        end
    otherwise
        disp('cannot be found')
end
% error_dis = [];
% for i = 2 : size(posenet_x,1)
% %     error_angle = [error_angle;angleDifference(Orient_est,realorientation)];
%     error_dis   = [error_dis;disDifference([posenet_x(i,1) posenet_y(i,1) posenet_z(i,1)],groundtruth.robotpose(i,:))];
% end

aviname     = ['SampleVideo_' subdataset '_posenet.avi'];
aviobj      = VideoWriter(aviname);
aviobj.FrameRate=5;
open(aviobj);


size_x = [min(posenet_x) max(posenet_x)];
size_y = [min(posenet_y) max(posenet_y)];
size_x = [size_x(1)-0.1*(size_x(2)-size_x(1)) size_x(2)+0.6*(size_x(2)-size_x(1))];
size_y = [size_y(1)-0.1*(size_y(2)-size_y(1)) size_y(2)+0.1*(size_y(2)-size_y(1))];
% for i = 2 : size(posenet_x,1)
for i = 2 :10: 3000
    createfigure([posenet_x(1:i,1) groundtruth.robotpose(1:i,1)],[posenet_y(1:i,1) groundtruth.robotpose(1:i,2)],[posenet_z(1:i,1) groundtruth.robotpose(1:i,3)],size_x,size_y);
 
    m = getframe(gcf);
    writeVideo(aviobj,m);
    close all;
end

close(aviobj);