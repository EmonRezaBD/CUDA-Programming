clear;
clc;
%image read
path = 'D:\\CUDA_WLI\\SFF\\Sukuna.JPG';
%path = 'D:\\CUDA_WLI\\SFF\\Data\\a1\\a1_50.BMP';
img = imread(path);

%convert to gray
img_gray = rgb2gray(img);
%convert to double
img_d = double(img_gray);

%keep copies of the original version
original_img1 = img_d;
original_img2 = img_d;

%Define Laplacian filter
Laplacian = [0 1 0; 1 -4 1; 0 1 0];
%Define Modified Laplacian Filter
modified_Laplacian = 1/16*[0 1 0; 1 -4 1; 0 1 0];

%Convolve the image using Laplacian filter
Laplacian_img = conv2(img_d, Laplacian, 'same');

%Convolve the image using Modified Laplacian filter
Modified_Laplacian_img = conv2(original_img1, modified_Laplacian, 'same');

%Define SML filter
M_h = [-1 2 -1; 0 0 0; 0 0 0];
M_v = [0 -1 0; 0 2 0; 0 -1 0];

% tic;
ML_3h = convn(original_img2,M_h,'same');
ML_3v = convn(original_img2,M_v,'same');
ML3 = abs(ML_3h) + abs(ML_3v);
% toc;

%To enhance the edges
N = 4;
sum_mask = ones(2*N+1);
SML3 = convn(ML3,sum_mask,'same'); %it's giving the edges in the boundary that means it has a sharp change there
%SML will work for stack of images where some images are clear and some are
%blurred. Then, we can analysis those edges for shape
%imtool(SML3)

% Create a figure to hold the subplots
figure;

% Display the first image
subplot(2, 2, 1); % 1 row, 3 columns, position 1
imshow(img);
title('Original Image');

% Display the second image
subplot(2, 2, 2); % 1 row, 3 columns, position 2
imshow(Laplacian_img);
title('After Laplacian filtering')

% Display the third image
subplot(2, 2, 3); % 1 row, 3 columns, position 3
imshow(Modified_Laplacian_img);
title('After Modified Laplacian filtering')

% Display the fourth image
subplot(2, 2, 4); % 1 row, 3 columns, position 3
imshow(SML3);
title('After SML filtering')

