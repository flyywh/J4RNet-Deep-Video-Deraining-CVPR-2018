dir_list = dir('*.png');

for i=1:length(dir_list)
    image = imread(dir_list(i).name);
    imwrite(image, [num2str(i,'%03d'), '.png']); 
end