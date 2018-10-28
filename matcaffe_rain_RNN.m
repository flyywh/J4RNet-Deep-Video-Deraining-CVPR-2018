function [net,result] = matcaffe_SRCNN_joint_rain(net, im)
input_data = reshape(im,size(im,1), size(im,2), size(im,3));
[h1,h2,h3,h4] = size(input_data);
input_final = input_data;
[height, width,ch, ~] = size(input_final);

net.blobs('data').reshape([height width ch 1]);
scores = net.forward({input_final});

result = scores{1};
