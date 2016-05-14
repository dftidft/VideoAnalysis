
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters
padding = 2;  % Extra area surrounding the target
dsift_bin_size = 4;
dsift_step = 8;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image file path
BASE_PATH = 'D:/Dataset/tracking/seq_bench/';
SEQ_NAME = 'motorrolling';
IMG_DIR = sprintf('%s/%s', BASE_PATH, SEQ_NAME);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get ground truth
GT_FILE_NAME = 'groundtruth_rect.txt';
gt_file_path = sprintf('%s/%s', IMG_DIR, GT_FILE_NAME);
gt_rects = importdata(gt_file_path);
pos = [gt_rects(1, 2) + gt_rects(1, 4) / 2, gt_rects(1, 1) + gt_rects(1, 3) / 2];
target_sz = [gt_rects(1, 4),  gt_rects(1, 3)];
win_sz = target_sz * (1 + padding);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for iframe = 1 : 1000
    % Read input
    img_file_path = sprintf('%s/img/%04d.jpg', IMG_DIR, iframe);
    if ~exist(img_file_path, 'file')
        fprintf('file does not exist\n');
        break;
    end
    
    img = imread(img_file_path);
    if ndims(img) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end
    
    if iframe == 1
        tic;
        [patch, offset] = crop_patch(gray, pos, target_sz);
        init_offset = offset;
        [keypoints, descriptors] = vl_dsift(single(patch), 'size', dsift_bin_size, 'step', dsift_step, 'fast');
        toc;
    else
        [patch, offset] = crop_patch(gray, pos, win_sz);
        [query_keypoints, query_descriptors] = vl_dsift(single(patch), 'size', dsift_bin_size, 'step', dsift_step, 'fast');
        [matches, scores] = vl_ubcmatch(descriptors, query_descriptors);
    end
    
    if iframe == 1
        fig1 = figure();
        fig2 = figure();
        figure(fig1);
        img_h = imshow(img, 'Border','tight', 'InitialMag', 100);
        hold on;
        figure(fig2);
        img_h2 = imshow(img, 'Border','tight', 'InitialMag', 100);
        hold on;
    else
        figure(fig1);
        set(img_h, 'CData', img);
    end
    
    % 在图中显示匹配点
    if iframe == 1
        figure(fig1);
        pts_h = plot(keypoints(1, :) + double(offset(2)), keypoints(2, :) + double(offset(1)), '.', 'Color', [1, 0, 0]);
        figure(fig2);
        pts_h2 = plot(keypoints(1, :) + double(offset(2)), keypoints(2, :) + double(offset(1)), '.', 'Color', [1, 0, 0]);
    else
        figure(fig1);
        set(pts_h, 'X', query_keypoints(1, matches(2, :)) + double(offset(2)));
        set(pts_h, 'Y', query_keypoints(2, matches(2, :)) + double(offset(1)));
        set(pts_h, 'Color', [0 0 1]);
        figure(fig2);
        set(pts_h2, 'X', keypoints(1, matches(1, :)) + double(init_offset(2)));
        set(pts_h2, 'Y', keypoints(2, matches(1, :)) + double(init_offset(1)));
        set(pts_h2, 'Color', [1 0 0]);
    end
    
    
    
    if iframe == 1
        text_h = text(20, 20, sprintf('%d', iframe), 'Color', [0 1 1], 'FontSize', 20);
    else
        set(text_h, 'string', sprintf('%d', iframe));
    end
    
    if double(get(gcf,'CurrentCharacter')) == 27
        break;
    end
    
    pause();

end

