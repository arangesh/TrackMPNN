% Function which propagates and adds noise to the 3D points and popoulates
% other variables of the structs passed to the function
% motion = [trans, rot_yaw_theta]
function detectionsQ = propagateDetectionsMod(detectionsQ, params_2D3D, params_carCuboid, motion)        
    
%% for all detections in Query propagate the box to 3D box in first frame  
    
    for i = 1:length(detectionsQ)                        
        
        b1Q = [];
        B1Q = [];
        bvolumeQ1 = [];
        
        % check if the detection is first time i.e. we have to start with
        % the canonical cuboid. Else we already have the cuboid.
        

        %disp(sprintf('first_time - id :%d\n',detectionsQ{i}.dno));
        canonicalCuboid = getCanonicalCuboid(params_carCuboid.avg_Sz); % [8, 3] storing 3d coordinates of each vertex in vehicle coord system
        
        % find center of the bounding box bottom line
        b1Q = [detectionsQ{i}.bbox(1) + (detectionsQ{i}.bbox(3) - detectionsQ{i}.bbox(1))/2 ;
            detectionsQ{i}.bbox(4);
            1.0];
        
        % project bottom center point to 3D
        B1Q = (params_2D3D.h * inv(params_2D3D.K) * b1Q) / (params_2D3D.n' * inv(params_2D3D.K) * b1Q); % (h K^(-1) x)/(n^T K^(-1) x)
        
        % apply offset which is a function of yaw and get car's origin (remember approx.)
        % computes offset to be applied to X i.e. center of bottom line of bbox of
        % detections to actually go to the center of the car. This offset should be
        % applied only in Z direction.
        offset_Z = getOffsetBasedOnYaw([params_carCuboid.avg_Sz(1); params_carCuboid.avg_Sz(3)], detectionsQ{i}.yaw);
        
        % car's origin in Query Frame
        B1Q = B1Q + [0; 0; offset_Z];
        
        % translate canonical cuboicameralink
        % i.e. get 3d corner locations of cuboid in camera coordinate
        % frame
        canonicalCuboid = canonicalCuboid + repmat(B1Q', 8,1);
        
        % BOUNDING VOLUME IN QUERY FRAME
        [bvolumeQ1, ~] = getBoundingVolume(B1Q, canonicalCuboid, detectionsQ{i}.yaw, ...
            detectionsQ{i}.sigma_3D, ...
            params_carCuboid.sz_ub, params_carCuboid.sz_lb ...
            );
        
        bvolumeQ1 = bvolumeQ1 - repmat([0 params_carCuboid.avg_Sz(2)/2 0], size(bvolumeQ1,1), 1);

        
        % car's origin in Train frame
        B2Q = motion(1:3) + B1Q ;
        bvolumeQ1 = bvolumeQ1 + repmat(motion(1:3)', size(bvolumeQ1,1),1);
        % CUBOID IN TRAIN FRAME
        
        %dummy_sigma = eye(4,4);
        %dumm_sigma(4,4) = 0;
        
        [bvolumeQ2, k2] = getBoundingVolume(B2Q, bvolumeQ1, motion(4), ... 
                                             detectionsQ{i}.sigma_3D, ...
                                             params_carCuboid.sz_ub, params_carCuboid.sz_lb ...
                                            );                 
        
       % bvolumeQ2 = bvolumeQ2 - repmat([0 params_carCuboid.avg_Sz(2)/2 0], size(bvolumeQ2,1), 1);    % offset the h/2 as car ar to be on road             
        
        % compute projection of bvolume and store in bvolume_proj variable
%         bvolume_proj = (params_2D3D.K*bvolumeQ2')';
%         bvolume_proj(:,1:3) = bvolume_proj(:,1:3)./repmat(bvolume_proj(:,3), 1,3);
%         kbvolume_proj = convhull(bvolume_proj(:,1:2));
%         bvolume_proj = bvolume_proj(kbvolume_proj,:);
        
        % update the fields of the structure
        detectionsQ{i}.bvolume = bvolumeQ2;
%         detectionsQ{i}.bvolume_proj = bvolume_proj;
        detectionsQ{i}.yaw  = detectionsQ{i}.yaw+motion(4);
        detectionsQ{i}.origin  = B2Q;
        detectionsQ{i}.k = k2;
        
        kbvolumeQ_xz = convhull(bvolumeQ2(:,1),  bvolumeQ2(:,3));
        detectionsQ{i}.bvolume_proj  = [bvolumeQ2(kbvolumeQ_xz,1) bvolumeQ2(kbvolumeQ_xz,3)];
    end
end