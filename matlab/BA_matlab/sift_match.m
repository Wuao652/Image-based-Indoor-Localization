Ia = imread(fullfile('vlfeat-0.9.21\data','roofs1.jpg')) ;
Ib = imread(fullfile('vlfeat-0.9.21\data','roofs2.jpg')) ;
Ia = single(rgb2gray(Ia));
Ib = single(rgb2gray(Ib));
[fa, da] = vl_sift(Ia) ;
[fb, db] = vl_sift(Ib) ;
[fa, da] = vl_sift(Ia,'EdgeThresh',20,'Octaves',4,'Levels',6) ;
[fb, db] = vl_sift(Ib,'EdgeThresh',20,'Octaves',4,'Levels',6) ;
[matches, scores] = vl_ubcmatch(da, db) ;

pts1 = [fa(1,matches(1,:));fa(2,matches(1,:))]';
pts2 = [fa(1,matches(2,:));fa(2,matches(2,:))]';