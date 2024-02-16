[filename, S, N, D, M] = read_corner_parameters('cornerparams.dat');

img = imread('/Users/kylechen/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/Senior/S24/CMPEN 454/cmpen-454-project1/Project1AssignmentFiles/checkerboard.jpg'); % Read the image
gray_img = double(rgb2gray(img)); % Convert to grayscale and to double precision

% Decide on the size of the Gaussian kernel based on sigma S
kernel_size = 2*ceil(2*S)+1; % A common choice
gauss_kernel = fspecial('gaussian', kernel_size, S);

% Smooth the image
smooth_img = imfilter(gray_img, gauss_kernel, 'same', 'replicate');

% Define finite difference kernels for x and y directions
dx = [-1 0 1; -1 0 1; -1 0 1]; % Example kernel for partial derivative in x
dy = dx'; % Transpose to get the kernel for y

% Compute gradients
Gx = imfilter(smooth_img, dx, 'same', 'replicate');
Gy = imfilter(smooth_img, dy, 'same', 'replicate');


%% Harris corner detection
% Compute products of derivatives
Gx2 = Gx.^2;
Gy2 = Gy.^2;
Gxy = Gx.*Gy;

% Define a box filter for the sums
box_filter = fspecial('average', N);

% Compute sums over the local N x N neighborhood
Sx2 = imfilter(Gx2, box_filter, 'same', 'replicate');
Sy2 = imfilter(Gy2, box_filter, 'same', 'replicate');
Sxy = imfilter(Gxy, box_filter, 'same', 'replicate');

% Compute the Harris corner "R" score values
k = 0.04; % Sensitivity factor, common choice
R = (Sx2.*Sy2 - Sxy.^2) - k*(Sx2 + Sy2).^2;

% Thresholding or non-maximum suppression to find corners would follow here,
% but since the prompt does not specify how to use D and M parameters for
% selecting the top M corners after non-max suppression, this part is omitted.
