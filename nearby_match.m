function [all_match, all_score] = nearby_match(query_keypoints, db_keypoints, query_descriptors, db_descriptors)

% Max distance between matching points
thres_dist = 32;
% Min ratio of (the best score / second best score)
thres_ratio = 1.5;

query_keypoints = int32(query_keypoints);
db_keypoints = int32(db_keypoints);
query_descriptors = int32(query_descriptors);
db_descriptors = int32(db_descriptors);

all_match = [];
all_score = [];
for i = 1 : size(query_keypoints, 2)
    dist = bsxfun(@minus, db_keypoints, query_keypoints(:, i));
    dist = sum(dist .^2);
    idx_dist_mask = find(dist < thres_dist ^2);
    score = bsxfun(@minus, db_descriptors, query_descriptors(:, i));
    score = sum(score .^2);
    [sorted_score, sorted_idx] = sort(score(idx_dist_mask));
    % [sorted_score, sorted_idx] = sort(score);
    best = sorted_score(:, 1);
    second_best = sorted_score(:, 2);
    if best * thres_ratio < second_best
        all_match = [all_match [i ; idx_dist_mask(sorted_idx(1))]];
        % all_match = [all_match [i ; sorted_idx(1)]];
        all_score = [all_score best];
    end
end

end