addpath('external/rp/matlab');
addpath('external/rp/cmex');
configFile = 'external/rp/config/rp_4segs.mat'; 
configParams = LoadConfigFile(configFile);

cnn_definition_file			= '/mnt/neocortex/scratch/tsechiw/SPP_net_old/data/cnn_model/Zeiler_conv5/Zeiler_spm_scale224_test_conv5.prototxt';
cnn_binary_file				= '/mnt/neocortex/scratch/tsechiw/SPP_net_old/data/cnn_model/Zeiler_conv5/Zeiler_conv5';
opts.spp_params_def         = fullfile(pwd, 'data', 'Zeiler_conv5_new', 'spp_config');
opts.cache_name             = 'Zeiler_conv5_ft(5s_flip)_fc7';

thresh = -1;

spp_model = spp_create_model(cnn_definition_file, cnn_binary_file, opts.spp_params_def, opts.cache_name);
opts.feat_cache             = 'Zeiler_conv5';
opts.gpu_id					= 3;
spp_model = spp_load_model(spp_model, true);
spp_ft_model = load('/mnt/neocortex3/scratch/yejiayu/SPP_net_intu/cachedir/Zeiler_conv5_ft(5s_flip)_fc7/intubation_train/spp_model');

im = imread('./example/images/07_1316.jpg');
%g = gpuDevice(2);

tic
dets = spp_detect(im, configParams, spp_model, spp_ft_model, thresh);
toc

%reset(g)