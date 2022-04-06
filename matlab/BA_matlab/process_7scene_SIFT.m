function [ observe_ith,param] = process_7scene_SIFT(history,cameraParams,observe_ith,filepath,filepath_history,num_image,gap,locID,locID_init,param_keyframe)
%GETIMAGE Summary of this function goes here


%   Select images for triangulation
ind_neighbor = selectimage(history.orientation,history.robotpose,locID_init,param_keyframe.bm,param_keyframe.sigma,param_keyframe.alpha_m,num_image,param_keyframe.max_range);

disp(ind_neighbor)

num_image_left = (num_image - 1) / 2;
nImgLst = cell(2*num_image_left+2,1);
nImgIndList = zeros(2*num_image_left+2,1);
nImgLst{1} = [filepath num2str(locID-1,'%010.0f') '.png'];
nImgIndList(1) = locID;

for i = 1 : size(ind_neighbor,1)
    nImgLst{i+1} = [filepath_history num2str(ind_neighbor(i)-1,'%010.0f') '.png'];
    nImgIndList(i+1) = ind_neighbor(i);
end

num_image = size(nImgLst,1) - 1;
IPrev = rgb2gray(imread(nImgLst{1}));
[featuresPrev,pointsPrev] = extractSIFTFeature(IPrev);
% pointsPrev = detectSURFFeatures(IPrev, 'MetricThreshold', 500,'NumScaleLevels',6,'NumOctaves',4);
Orient = observe_ith.orientation;
robotpose = observe_ith.robotpose;
viewID = [];
vSet = viewSet;
vSet = addView(vSet, 1,'Points',pointsPrev,'Orientation',...
    Orient',...
    'Location',robotpose);
for viewId = 2:num_image+1
    if(~exist(nImgLst{viewId},'file'))
        continue;
    end
    I = imread(nImgLst{viewId});
    I = rgb2gray(I);

    [features,points] = extractSIFTFeature(I);
    Orient = history.orientation(3*nImgIndList(viewId)-2:3*nImgIndList(viewId),:);
    robotpose = history.robotpose(nImgIndList(viewId),:);

   % pairsIdx = matchFeatures(featuresPrev,features,'Method','Approximate','MaxRatio',0.8,'Unique',true);
    [pairsIdx, scores] = vl_ubcmatch(featuresPrev', features');pairsIdx = pairsIdx';
    
    %   Add RANSAC
    matchedPoints1 = pointsPrev(pairsIdx(:,1),:);
    matchedPoints2 = points(pairsIdx(:,2),:);
%     if(matchedPoints1.Count > 20)
    if(size(matchedPoints1,1) > 20)
        vSet = addView(vSet,viewId,'Points',points,'Orientation',...
        Orient','Location',robotpose);
        %[fLMedS, inliers] = estimateFundamentalMatrix(matchedPoints1.Location,matchedPoints2.Location,'NumTrials',2000);
        [fLMedS, inliers] = estimateFundamentalMatrix(matchedPoints1,matchedPoints2,'NumTrials',2000);

        pairsIdx = pairsIdx(inliers,:);
        
        vSet = addConnection(vSet,1,viewId,'Matches',pairsIdx);
        viewID = [viewID;viewId];
    end  
end

if(size(vSet.Views,1) <  num_image_left+1)
    observe_ith = [];
    param = [];
    return
end
%   Find 3-D world points.
[tracks1,tracks2] = create_Tracks(vSet,num_image+1);
cameraPoses = poses(vSet,viewID);
if(isempty(tracks2))
    Orient = observe_ith.orientation;
    robotpose = observe_ith.robotpose;
    disp('No SURF features are found');
    return
end
[xyzPoints,errors] = triangulateMultiview(tracks2,cameraPoses,cameraParams);

% figure()
% scatter3(xyzPoints(:, 1), xyzPoints(:, 2), xyzPoints(:, 3))


idx = errors < 5;


%   Filter outlier
xyzPoints = xyzPoints(idx,:);
tracks1 = tracks1(1,idx);
observe_ith.pts3D = xyzPoints;
observe_ith.pts2D = zeros(size(xyzPoints,1),2);
param.K = cameraParams.IntrinsicMatrix';
param.num_features = size(xyzPoints,1);
for i =1 : size(tracks1,2)
    observe_ith.pts2D(i,:) = [tracks1(1,i).Points(1,2) tracks1(1,i).Points(1,1)];
end


end

