function dets = spp_detect(im, configParams, spp_model, spp_ft_model, thresh)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

fprintf('Computing candidate regions...');
th = tic();
boxes = RP(im, configParams);
fprintf('found %d candidates (in %.3fs).\n', size(boxes,1), toc(th));

% extract features from candidates (one row per candidate box)
fprintf('Extracting CNN features from regions...');
th = tic();

feat_cache = [];
opts.spm_im_size = [480 576 688 874 1200];
% convert image to feature output by conv5 layer
d.feat = spp_features_convX(im, opts.spm_im_size, feat_cache, true);

d.feat = spp_features_convX_to_poolX(spp_model.spp_pooler, d.feat, boxes);
%% from here, use fine-tune network
spp_model = spp_ft_model.spp_model;
spp_model.cnn.layers = spp_layers_in_gpu(spp_model.cnn.layers);
d.feat = spp_poolX_to_fcX(d.feat, spp_model.training_opts.layer, spp_model, true);

d.feat = spp_scale_features(d.feat, spp_model.training_opts.feat_norm_mean);
fprintf('done (in %.3fs).\n', toc(th));

% compute scores for each candidate [num_boxes x num_classes]
fprintf('Scoring regions with detectors...');
th = tic();
scores = bsxfun(@plus, spp_model.detectors.W * d.feat, spp_model.detectors.B)';
fprintf('done (in %.3fs)\n', toc(th));

% apply NMS to each class and return final scored detections
fprintf('Applying NMS...');
th = tic();
num_classes = length(spp_model.classes);
dets = cell(num_classes, 1);
for i = 1:num_classes
  I = find(scores(:, i) > thresh);
  scored_boxes = cat(2, boxes(I, :), scores(I, i));
  keep = nms(scored_boxes, 0.3); 
  dets{i} = scored_boxes(keep, :);
end
fprintf('done (in %.3fs)\n', toc(th));
reset(g)
