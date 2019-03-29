clear all;
close all;
clc;

%InputPath='E:\matlabFiles\LBF-active-contour-model-master\rename_good\';
InputPath='E:\matlabFiles\LBF-active-contour-model-master\rename_bad\';
FileName=dir(strcat(InputPath,'*.jpg'));
NumFile=length(FileName); %输出数据类型为double
%fprintf(class(NumFile)); %结果为double
for img_index=1:NumFile
    tempFileName=FileName(img_index).name;
    ImPath=strcat(InputPath,tempFileName);
    Img= imread(ImPath); 
    if size(Img,3) == 3
        Img = rgb2gray(Img);
    end
    I = double(Img);
    [r, c] = size(I);
    phi = ones(r, c) .* -2;
    
    %% Hyper Parameters and randomly generate initial location
    Eps = 1; 
    eta = 0.1;
    Kernel_Sigma = 3;
    Iteration=120;  
    L1=1;  L2=1; 
    mu=1;    
    nu=0.001*255^2;

    boardsize = 10; %距离边界的位置
    r = 10; %产生圆形时为半径，产生矩形时为(1/2)*边长
    if r > boardsize
        r = boardsize;
    end
    possiblex = (boardsize + 1): (size(I,1) - boardsize);
    possibley = (boardsize + 1): (size(I,2) - boardsize);
    labelx = randperm(length(possiblex));
    labely = randperm(length(possibley));
    centrex = possiblex(labelx(1));
    centrey = possibley(labely(1));
    %[m,n] = size(Img);
    %phi= -ones(m,n).*c0;
   %产生矩形
    for x = 1:size(I,1)
        for y = 1:size(I,2)
            phi(centrex-r:centrex+r,centrey - r:centrey + r) = 2;
        end
    end
    %%

    K = fspecial('gaussian',  1 + 4 * Kernel_Sigma, Kernel_Sigma);

    for i = 1 : Iteration

        H_eps = (1 + (2/pi) * atan(phi ./ Eps)) / 2;
        Delta_eps = (1 / pi) .* (Eps ./ (Eps^2 + phi.^2));

        F1 = conv2(I .* H_eps , K , 'same') ./ conv2(H_eps , K , 'same');
        F2 = conv2(I .* (1 - H_eps) , K , 'same') ./ conv2((1 - H_eps) , K , 'same');

        T_Region = -Delta_eps .* (I.^2 .* (L1 - L2) - 2 * I .* conv2((L1 * F1 - L2 * F2), K, 'same') + conv2((L1 * F1 .^ 2 - L2 * F2 .^ 2), K, 'same'));
        T_Regulator = nu .* Delta_eps .* kappa(phi) + mu .* (del2(phi) - kappa(phi));

        phi = phi + eta .* (T_Region + T_Regulator);

        if(mod(i,20)==0)
            imshow(Img,[],'initialmagnification','fit');
            hold on;
            contour(phi,[0 0],'g','LineWidth',1);
            title(strcat('Iteration: ', num2str(i))); 
            drawnow;
        end
    end

    result = phi <= 0;

%     subplot(2,2,4); 
%     imshow(result); 
%     title('Segmentation');

    imwrite(result,['E:\matlabFiles\LBF-active-contour-model-master\LBF_bad\',tempFileName]);
    %imwrite(result,['E:\matlabFiles\LBF-active-contour-model-master\LBF_good\',tempFileName]);
end