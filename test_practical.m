clear all;
warning('off');

use_gpu = 2;
gpu_id = 0;

if exist('/home/yangwenhan/Platform/caffe/matlab/+caffe', 'dir')
    addpath('/home/yangwenhan/Platform/caffe/matlab/');
else
    error('Please run this demo from caffe/matlab/demo');
end

model_dir = './SRCNN_model/';

save_tag = 'pracical_results';
net_model_path = [model_dir, 'deploy_rain_removal_BRCN_type.prototxt'];
net_weights = [model_dir, 'rain_removal_BRCN_type_iter_90000.caffemodel'];

phase = 'test'; 

if exist('use_gpu', 'var') && use_gpu
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
else
    caffe.set_mode_cpu();
end

if ~exist(net_weights, 'file')
    error('Please download CaffeNet from Model Zoo before you run this demo');
end

caffe.reset_all();
net = caffe.Net(net_model_path, net_weights, phase);

save_dir = 'results';
input_dir = 'inputs';

psnr_res = 0;
ssim_res = 0;
psnr_list  =zeros(100, 1);
ssim_list  =zeros(100, 1);


for i=1:50
    tmp_img = imread([input_dir, '/', num2str(i, '%03d'), '.png']);
    [hei, wid, ~] = size(tmp_img);
    
    tmp_cnt = 1;
    image_list = zeros(hei, wid, 9);
    for t=i-4:i+4
        image = imread([input_dir, '/', num2str(i, '%03d'), '.png']);
        
        image = rgb2ycbcr(image);
        if t==i
            image_cbcr = image(:, :, 2:3);
        end
        image = image(:,:,1);

        rain_image = double(image)/255;
        image_list(:,:,tmp_cnt) = rain_image;
        tmp_cnt = tmp_cnt+1;
    end
    
    tic;
    [im_res] = matcaffe_rain_RNN_one_direction(net, image_list);
    toc;
    
    im_res_final = cat(3, uint8(im_res(:,:,5)*255), image_cbcr(:, :, 1), image_cbcr(:, :, 2));
    im_res_final = ycbcr2rgb(im_res_final);
    
    output_image_name = ['./', save_dir, '/results-', num2str(i, '%03d'), '.png'];
    imwrite(im_res_final, output_image_name);
    
end


