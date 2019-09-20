% function to generate a cuboid based on the sz:[l h b], ry, corresponding
% noise and the origin. sz_ub/_lb are the upper and lower bounds on the
% size of the cuboid afte all the operations have gone in to it.
% RETURNS: bounding volume (bvolume) which is the convex hull of the points
% got after applying the yaw and lbh error to the avg car cuboid
% bounds have been applied.
% NOTE: - ry is +ve for counter-clockwise
%       - cuboid is created with facing towards X axis on the X-Z plane i.e.
%         length is in X axis, width in Z axis and Y is for height starting
%         from 0 to sz(y)
        

function [bvolume, k] = getBoundingVolume(center, pts, ry, sigma_3D, sz_ub, sz_lb)

    % 3D points stacked after applying the -ve and +ve yaw error
    
    ry_N = sigma_3D(4,4);        
    centered_pts = pts-repmat(center', size(pts,1),1);  % bring pts to origin to rotate
    
    rot_pts_plus_yaw  = (rotMatY_3D(+ry_N) * centered_pts'  )';
    rot_pts_minus_yaw = (rotMatY_3D(-ry_N) * centered_pts'  )';
    
    pts = [centered_pts;
           rot_pts_plus_yaw;
           rot_pts_minus_yaw  ]; 
    
    % get the convex hull of all pts i.e. after error in yaw
    k = convhull(pts(:,1), pts(:,2), pts(:,3));
    bvolume = pts(k, :);  
    
    % scale the bounding voluem based on the error in the three axis i.e.
    % in l, h, w
    scales = diag(sigma_3D(1:3, 1:3));
    scale_mat = [scales(1) 0 0; 0 scales(2) 0; 0 0 scales(3)];
    bvolume = (scale_mat * bvolume')';
           
    % bound the bvolume by bounds in l,b,and h          
    % apply upper and lower bounds on the newly formed outer bound cuboid
 
    % bound in X
    bvolume(find(bvolume(:,1)>sz_ub(1)), 1) = sz_ub(1);
    bvolume(find(bvolume(:,1)<sz_lb(1)), 1) = sz_lb(1);
    
    % bound in Y
    bvolume(find(bvolume(:,2)>sz_ub(2)), 2) = sz_ub(2);
    bvolume(find(bvolume(:,2)<sz_lb(2)), 2) = sz_lb(2);
    
    % bound in Z
    bvolume(find(bvolume(:,3)>sz_ub(3)), 3) = sz_ub(3);
    bvolume(find(bvolume(:,3)<sz_lb(3)), 3) = sz_lb(3);           
    
    % finally apply the yaw angle of the car to the bvolume
    bvolume = (rotMatY_3D(ry) * bvolume');
    
    %+ repmat(center, 1,size(bvolume,1)))';  
    
    % re-translate the point to its true origin
    bvolume = (bvolume + repmat(center, 1,size(bvolume,2)))';
    
    k = convhull(bvolume);
end
