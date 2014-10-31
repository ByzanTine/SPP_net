%cnn_definition_file 		= '/mnt/neocortex3/scratch/yejiayu/SPP_net_intu/finetuning/Zeiler_conv5_ft(5s_flip)/pascal_finetune_fc_spm_train_test_new.prototxt';
%cnn_binary_file 			= '/mnt/neocortex3/scratch/yejiayu/SPP_net_intu/finetuning/Zeiler_conv5_ft(5s_flip)/FT_iter_12000';
cnn_definition_file			= '/mnt/neocortex/scratch/tsechiw/SPP_net_old/data/cnn_model/Zeiler_conv5/Zeiler_spm_scale224_test_conv5.prototxt';
cnn_binary_file				= '/mnt/neocortex/scratch/tsechiw/SPP_net_old/data/cnn_model/Zeiler_conv5/Zeiler_conv5';
opts.spp_params_def         = fullfile(pwd, 'data', 'Zeiler_conv5_new', 'spp_config');
opts.cache_name             = 'Zeiler_conv5_ft(5s_flip)_fc7';

spp_model = spp_create_model(cnn_definition_file, cnn_binary_file, opts.spp_params_def, opts.cache_name);
opts.feat_cache             = 'Zeiler_conv5';
spp_model = spp_load_model(spp_model, true);

im = imread('./example/images/000084.jpg');

feat_cache = [];
opts.spm_im_size = [480 576 688 874 1200];
% convert image to feature output by conv5 layer
d.feat = spp_features_convX(im, opts.spm_im_size, feat_cache, conf.use_gpu);

% convert conv_feature to pooling feature
% d.feat = spp_features_convX_to_poolX
% convert pool_feature to fc feature
d.feat = spp_poolX_to_fcX(d.feat, feat_opts.layer, spp_model, conf.use_gpu);
d.feat = spp_scale_features(d.feat, feat_opts.feat_norm_mean);

score = bsxfun(@plus, spp_model.detectors(f).W * d.feat, spp_model.detectors(f).B)';

%dets = spp_detect(im, spp_model, -1);