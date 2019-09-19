addpath(genpath('./third_party/cocoapi/MatlabAPI'));

warning('off','all')
clc;
clear ;
root = '/path/to/kitti-mots';

dataset_path = fullfile(root, 'testing');
mkdir(fullfile(dataset_path, 'gcn_features'));

% road plane normal and camera height
n =  [0; 1; 0]; % normal vector in camera coordinates
h =  1.72; % height of camera in m

% average car parameters in meters [l, h, w];
avgCar_Sz = [4.3; 2; 2];
sz_ub     = [34; 31.5; 31.5]; % size upper bound
sz_lb     = [-34; -31.5; -31.5]; % size lower bound

K_all     = load(fullfile(dataset_path, 'calib', 'calib_all.txt')); % camera matrix for all train sequences (one for each drive)


%%
for s = 0:28
    seqNo = s;
    mkdir(fullfile(dataset_path, 'gcn_features', sprintf('%.4d', seqNo)));
    
    %% SETUP PARAMETERS
    pose_path  = fullfile(dataset_path, 'ORBSLAM_pose', sprintf('%04d', seqNo), 'KITTITrajectoryComplete_new');  % pose path for current sequence
    K          = [ K_all(seqNo+1,1:3); K_all(seqNo+1,5:7); K_all(seqNo+1,9:11)]; % intrinsics for current sequence
    
    
    % struct which holds the basic attributes of an avg. car
    params_carCuboid = struct('avg_Sz',         avgCar_Sz,    ...
        'sz_ub',          sz_ub,        ...
        'sz_lb',          sz_lb         ...
        );
    
    params_2D3D      = struct('K',              K,            ...
        'n',              n,            ...
        'h',              h             ...
        );
    
    detection       = struct('dno',            -1.0,         ...
        'score',          -1.0,         ...
        'bbox',           zeros(4,1),   ...
        'sigma_2D',       zeros(3,3),   ...
        'yaw',            0.0,          ...
        'bvolume',        [],           ...
        'k',              [],           ...
        'origin',         zeros(3,1),   ...
        'start_frame',    -1.0,         ...
        'end_frame',      -1.0,         ...
        'sigma_3D',       zeros(4,4)    ...-
        );
    
    %% LOAD DATA :
    % load camera pose (Monocular ORBSLAM)
    pose = load(pose_path); % load pose for entire seequence
    
    %load all detections cell array - variable name 'detections'
    % format: [x1, y1, x2, y2, score, track?]
    %all detections belong to class 'Car'
    load(fullfile(dataset_path, 'RRC_Detections_mat', sprintf('%04d', seqNo), 'detections.mat'));
    
    % feautes_2D2D. These features were extracted from our Keypoint Network
    load(fullfile(dataset_path, 'Features2D_mat', sprintf('%04d', seqNo), 'features_2_2D2D.mat'));
    
    
    %%
    for i = 1:size(pose,1)
        fprintf('Seq<%02d> | frame : %04d\n', seqNo, i);
        feats.score = {[]};
        feats.bbox_2d = {[]};
        feats.appearance = {[]};
        feats.convex_hull_3d = {[]};
        json_output = fullfile(dataset_path, 'gcn_features', sprintf('%.4d', seqNo), sprintf('%.6d.json', i-1));
        
        %%
        if ~isempty(detections{i})            
            %%
            num_detectionsQ = size(detections{i},1); % get num dets in current frame
            detectionsQ = cell(num_detectionsQ, 1); % initialize cell to store det structs in current frame
            
            % load the detections in to the query and train struct cell arrays x1 y1 x2 y2 confidence ID
            for j = 1:num_detectionsQ
                detectionsQ{j}                  = detection; % emtpy detection struct
                detectionsQ{j}.bbox             = detections{i}(j,1:4);
                detectionsQ{j}.score            = detections{i}(j,5);
                detectionsQ{j}.dno              = 0;
                detectionsQ{j}.yaw              = deg2rad(-90) ;
                detectionsQ{j}.sigma_3D         = [1.3 0 0 0; 0 1.1 0 0; 0 0 1.1 0; 0 0 0 deg2rad(0)];
            end
            
            %%
            % extract relative pose between current and FIRST FRAME!
            T = reshape(pose(i, 2:end), [3, 4]);
            r = rodrigues(T(1:3, 1:3));
            motion = [-(pose(1,11)-pose(i,11)); % translation params only?
                -(pose(1,12)-pose(i,12));
                -(pose(1,13)-pose(i,13));
                deg2rad(0)];
                %r(2)];
            
            % scale the motion using empiracally estimated scale factor (because ORBSLAM gives reconstructs upto scale)
            motion(1:3,1) =  motion(1:3,1).*(1.72/44);
            
            % propagat detections from current frame to next with
            % uncertainty in motion and heading
            detectionsQ = propagateDetections(detectionsQ,       ...
                params_2D3D,       ...
                params_carCuboid,  ...
                motion             ...
                );
            
            %%
            for j = 1:num_detectionsQ
                if j == 1
                    feats.score{end} = detectionsQ{j}.score;
                    feats.bbox_2d{end} = detectionsQ{j}.bbox;
                    app_feat = reshape(features_2D2D{i}(j, 2:end), [64, 64])';
                    app_feat = app_feat(1:8:end, 1:8:end); % subsample appearance feature channel
                    feats.appearance{end} = app_feat(:)';
                    feats.convex_hull_3d{end} = detectionsQ{j}.bvolume_proj;
                else
                    feats.score{end+1} = detectionsQ{j}.score;
                    feats.bbox_2d{end+1} = detectionsQ{j}.bbox;
                    app_feat = reshape(features_2D2D{i}(j, 2:end), [64, 64])';
                    app_feat = app_feat(1:8:end, 1:8:end); % subsample appearance feature channel
                    feats.appearance{end+1} = app_feat(:)';
                    feats.convex_hull_3d{end+1} = detectionsQ{j}.bvolume_proj;
                end
            end
            feats.score = feats.score';
            feats.bbox_2d = feats.bbox_2d';
            feats.appearance = feats.appearance';
            feats.convex_hull_3d = feats.convex_hull_3d';
        end
        
        %%
        f = fopen(json_output, 'w');
        fwrite(f, gason(feats));
        fclose(f);
    end
    fprintf('Done with sequence %d/%d...\n', seqNo, 28);
end