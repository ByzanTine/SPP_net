%cnn_definition_file 		= '/mnt/neocortex3/scratch/yejiayu/SPP_net_intu/finetuning/Zeiler_conv5_ft(5s_flip)/pascal_finetune_fc_spm_train_test_new.prototxt';
%cnn_binary_file 			= '/mnt/neocortex3/scratch/yejiayu/SPP_net_intu/finetuning/Zeiler_conv5_ft(5s_flip)/FT_iter_12000';
cnn_definition_file			= '/mnt/neocortex/scratch/tsechiw/SPP_net_old/data/cnn_model/Zeiler_conv5/Zeiler_spm_scale224_test_conv5.prototxt';
cnn_binary_file				= '/mnt/neocortex/scratch/tsechiw/SPP_net_old/data/cnn_model/Zeiler_conv5/Zeiler_conv5';
opts.spp_params_def         = fullfile(pwd, 'data', 'Zeiler_conv5_new', 'spp_config');
opts.cache_name             = 'Zeiler_conv5_ft(5s_flip)_fc7';

spp_model = spp_create_model(cnn_definition_file, cnn_binary_file, opts.spp_params_def, opts.cache_name);
opts.feat_cache             = 'Zeiler_conv5';
opts.gpu_id					= 3;
spp_model = spp_load_model(spp_model, true);

im = imread('./example/images/000084.jpg');

%dets = spp_detect(im, spp_model, -1);