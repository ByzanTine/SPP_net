
cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

fprintf('Downloading SPP_net_release1_data...\n');
urlwrite('https://onedrive.live.com/download?resid=4006CBB8476FF777!9723&authkey=!APTWXLD_P7UN6P0&ithint=file%2czip', ...
    'SPP_net_release1_data_for_new_caffe.zip');
fprintf('Unzipping...\n');
unzip('SPP_net_release1_data_for_new_caffe.zip', '../');

fprintf('Done.\n');
%system('del SPP_net_release1_data.zip');

cd(cur_dir);
