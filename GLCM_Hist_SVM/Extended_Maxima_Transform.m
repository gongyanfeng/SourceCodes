clear all;
close all;
clc;

InputPath='E:\matlabFiles\bullet_photo\';
OutputPath_bad='E:\matlabFiles\bullet_photo\bad_emt\';
OutputPath_good='E:\matlabFiles\bullet_photo\good_emt\';

subdirName=dir(InputPath);
Numdir=length(subdirName); %输出数据类型为double
disp(Numdir)
for i=1:Numdir
    if strcmp(subdirName(i).name,'bad')
        FileName=dir(strcat(InputPath,'bad\','*.jpg'));
        disp(FileName)
        NumFile=length(FileName); %输出数据类型为double
        disp(NumFile)
        for j=1:NumFile
            tempFileName=FileName(j).name;
            ImBadPath=strcat(InputPath,'bad\',tempFileName);
            im= imread(ImBadPath); 
            IEMT = imextendedmax(im,20);
            imwrite(IEMT,[OutputPath_bad,tempFileName]);
        end
    elseif strcmp(subdirName(i).name,'good')
        FileName=dir(strcat(InputPath,'good\','*.jpg'));
        disp(FileName)
        NumFile=length(FileName); %输出数据类型为double
        disp(NumFile)
        for j=1:NumFile
            tempFileName=FileName(j).name;
            ImGoodPath=strcat(InputPath,'good\',tempFileName);
            im= imread(ImGoodPath); 
            IEMT = imextendedmax(im,20);
            imwrite(IEMT,[OutputPath_good,tempFileName]);
        end
    end
end