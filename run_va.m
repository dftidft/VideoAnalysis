%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Just match the points nearby to avoid the two wheels/eyes problem
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters
padding = 2;  % Extra area surrounding the target
dsift_bin_size = 4;
dsift_step = 4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image file path
BASE_PATH = 'd:/dataset/tracking/seq_bench/';
SEQ_NAME = 'woman';
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
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Init(train) or predict
    if iframe == 1
        [patch, offset] = crop_patch(gray, int32(pos), int32(target_sz));
        init_offset = offset;
        tic;
        [keypoints, descriptors] = vl_dsift(single(patch), 'size', dsift_bin_size, 'step', dsift_step, 'fast');
        toc;
        keypoints = bsxfun(@plus, keypoints, double([init_offset(2); init_offset(1)]));
    else
        [patch, offset] = crop_patch(gray, int32(pos), int32(win_sz));
        [candi_keypoints, candi_descriptors] = vl_dsift(single(patch), 'size', dsift_bin_size, 'step', dsift_step, 'fast');
        candi_keypoints = bsxfun(@plus, candi_keypoints, double([offset(2); offset(1)]));
        tic;
        % [matches, scores] = nearby_match(keypoints, candi_keypoints, descriptors, candi_descriptors);
        [matches, scores] = nearby_match_fast(keypoints, candi_keypoints, descriptors, candi_descriptors);
        matches = uint32(matches);
        toc;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Display
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
        figure(fig1);
        set(img_h2, 'CData', prev_img);
    end
    
    if iframe == 1
        figure(fig1);
        pts_h = plot(keypoints(1, :), keypoints(2, :), '.', 'Color', [1, 0, 0]);
        figure(fig2);
        pts_h2 = plot(keypoints(1, :), keypoints(2, :), '.', 'Color', [1, 0, 0]);
    else
        figure(fig1);
        set(pts_h, 'XData', candi_keypoints(1, matches(2, :)));
        set(pts_h, 'YData', candi_keypoints(2, matches(2, :)));
        set(pts_h, 'Color', [0 0 1]);
        figure(fig2);
        set(pts_h2, 'XData', keypoints(1, matches(1, :)));
        set(pts_h2, 'YData', keypoints(2, matches(1, :)));
        set(pts_h2, 'Color', [1 1 0]);
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
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Update
    prev_pos = pos;
    if iframe > 1
        pos = [gt_rects(iframe, 2) + gt_rects(iframe, 4) / 2, gt_rects(iframe, 1) + gt_rects(iframe, 3) / 2];
        target_sz = [gt_rects(iframe, 4),  gt_rects(iframe, 3)];
        win_sz = target_sz * (1 + padding);
        [patch, offset] = crop_patch(gray, int32(pos), int32(target_sz));
        init_offset = offset;
        [keypoints, descriptors] = vl_dsift(single(patch), 'size', dsift_bin_size, 'step', dsift_step, 'fast');
        keypoints = bsxfun(@plus, keypoints, double([init_offset(2); init_offset(1)]));
    end
    prev_img = img;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    

end

