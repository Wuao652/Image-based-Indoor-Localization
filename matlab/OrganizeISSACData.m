%   This is the code to Organize the ISSAC sim data set

%   Parameter setting
folderpath = 'E:\dataset\NVIDIA\indoor_dataset';

if(exist([folderpath '\posenet_training_output'],'dir'))
    rmdir([folderpath '\posenet_training_output'],'s');
    mkdir([folderpath '\posenet_training_output']);
    mkdir([folderpath '\posenet_training_output\cdf']);
    mkdir([folderpath '\posenet_training_output\logs']);
    mkdir([folderpath '\\posenet_training_output\training_data_info']);
    mkdir([folderpath '\posenet_training_output\weights']);
end
% if(exist([folderpath '\sequences'],'dir'))
%     rmdir([folderpath '\sequences'],'s');
    mkdir([folderpath '\sequences']);
% end
if(exist([folderpath '\tfrecord2'],'dir'))
    rmdir([folderpath '\tfrecord2'],'s');
    mkdir([folderpath '\tfrecord2']);
end

% I. Rename files
convertpose_fromISSAC([folderpath '\01_long_test.txt'],[folderpath '\01.txt']);
convertpose_fromISSAC([folderpath '\02_long_test.txt'],[folderpath '\02.txt']);
copyfile([folderpath '\01_long\img'],[folderpath '\sequences\00']);
copyfile([folderpath '\02_long\img'],[folderpath '\sequences\01']);



function [] = convertpose_fromISSAC(filename_src,filename_tar)
delete(filename_tar);
fid = fopen(filename_src);
fid1=fopen(filename_tar,'wt');

i = 1;
while 1    
    tline = fgetl(fid);
    if ~ischar(tline),   break,   end  % exit at end of file 
    ln = sscanf(tline,'%s',1); % line type
    if(isempty(ln))
        break;
    end

    mtl_name = split(tline);
    history.robotpose(i,:) = [-str2double(mtl_name{3,1}) str2double(mtl_name{4,1}) str2double(mtl_name{5,1})];

    tmp = [str2double(mtl_name{9,1}) -str2double(mtl_name{6,1}) str2double(mtl_name{7,1}) str2double(mtl_name{8,1}) ];
    history.orientation(3*i-2:3*i,:)=quat2rotm(tmp);
    
    linetxt{1} = mtl_name{1,1}(5:end);
    linetxt{2} = num2str(-str2double(mtl_name{3,1}));
    linetxt{3} = mtl_name{4,1};
    linetxt{4} = mtl_name{5,1};
    linetxt{5} = num2str(-str2double(mtl_name{6,1}));
    linetxt{6} = mtl_name{7,1};
    linetxt{7} = mtl_name{8,1};
    linetxt{8} = mtl_name{9,1};
    fprintf(fid1,'%s\n',strjoin(linetxt));
    i = i + 1; 
end
fclose(fid);
fclose(fid1);
end
