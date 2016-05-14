function [patch, offset] = crop_patch(img, pos, win_sz)

img_sz = size(img);
img_sz = img_sz(1:2);

top = max(1, pos(1) - win_sz(1) / 2);
left = max(1, pos(2) - win_sz(2) / 2);
bottom = min(img_sz(1), pos(1) + win_sz(1) / 2);
right = min(img_sz(2), pos(2) + win_sz(2) / 2);

patch = img(top:bottom, left:right, :);
offset = [top - 1, left - 1];

end