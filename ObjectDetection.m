clear;
close all

% load all the images into dataset
%global apple;

D = dir(fullfile('Apple_test','*JPG'));
fprintf('loading %02d',length(D));

%apple = cell(21,1);

for i = 1:length(D)
    fprintf('\b\b%02d',i);
end
fprintf('\n');

%clear D;
% end of load images



%function openbw=ObjectDetection()

mean_list = zeros(4,length(D));  % create mean values for each apple slice
% loop over all the images in the folder
for k = 1:length(D)
    src = imread(D(k).name);
    %src = imread('1_1600_1.JPG');
    row = size(src,1);
    col = size(src,2);
    src1 = rgb2gray(src); % convert the original image into black/white

    src2 = medfilt2(src1,[5 5]); %convolution with image with kernel size 5*5
    %imshow(src2)

    level = graythresh(src2); %calculating threshold
    bw = im2bw(src,level); %binarization image
    %imshow(bw)


    se = strel('square',2); %create a 2*2 square
    openbw = imopen(bw,se); %smooth the edge 

    stats = regionprops(openbw,'basic'); %get Area,Centroid,BoundingBox of the connected image

    centroids = cat(1,stats.Centroid);


    area = cat(1,stats.Area);
    area = sort(area,'descend');
    apple1 = area(2); %first apple
    apple2 = area(3); %second apple
    apple3 = area(4); %third apple
    apple4 = area(5); %forth apple
    centroidsOfficiall = [];
    boundingBox = [];
    list = [];


    for i=1:size(stats) 
        if stats(i).Area==apple1 | stats(i).Area==apple2 |stats(i).Area==apple3 |stats(i).Area==apple4
         %centroidsOfficiall = [centroidsOfficiall,stats(i).Centroid]
         %   boundingBox = [[boundingBox] [stats(i).BoundingBox]]
         list = [list,i];
        end
    end


    imshow(openbw),title('centroids')
    hold on

    %plot(centroids(:,1),centroids(:,2),'b*'),

    for i=1:size(list,2)
        a = stats(list(i)).Centroid;
        plot(a(1),a(2),'b*');
        rectangle('Position',stats(list(i)).BoundingBox,'LineWidth',2,'LineStyle','--','EdgeColor','r');

        % get five target windows around the centroids
        ax = round(a(1)-50);
        ay = round(a(2)-50);
        rectangle('Position', [ax ay 101 101], 'LineWidth',1,'LineStyle','-','EdgeColor','r');
        aax = ax - 101;
        aay = ay;
        rectangle('Position', [aax aay 101 101], 'LineWidth',1,'LineStyle','-','EdgeColor','r');
        abx = ax + 101;
        aby = ay;
        rectangle('Position', [abx aby 101 101], 'LineWidth',1,'LineStyle','-','EdgeColor','r');
        alx = ax;
        aly = ay - 101;
        rectangle('Position', [alx aly 101 101], 'LineWidth',1,'LineStyle','-','EdgeColor','r');
        arx = ax;
        ary = ay + 101;
        rectangle('Position', [arx ary 101 101], 'LineWidth',1,'LineStyle','-','EdgeColor','r');

        std_list = [];
        if max(ax,row)==row & min(ax-101,1)==1 & max(ay+101,col)==col & min(ay,1)==1
            src_m = src(((ax-101):ax), (ay:ay+101));
            std_list = [std_list std2(src_m)];
        end
        if max(aax,row)==row & min(aax-101,1)==1 & max(aay+101,col)==col & min(aay,1)==1
            src_a = src(((aax-101):aax), (aay:aay+101));
            std_list = [std_list std2(src_a)];
        end
        if max(abx,row)==row & min(abx-101,1)==1 & max(aby+101,col)==col & min(aby,1)==1
            src_b = src(((abx-101):abx), (aby:aby+101));
            std_list = [std_list std2(src_b)];
        end
        if max(alx,row)==row & min(alx-101,1)==1 & max(aly+101,col)==col & min(aly,1)==1
            src_l = src(((alx-101):alx), (aly:aly+101));
            std_list = [std_list std2(src_l)];
        end
        if max(arx,row)==row & min(arx-101,1)==1 & max(ary+101,col)==col & min(ary,1)==1
            src_r = src(((arx-101):arx), (ary:ary+101));
            std_list = [std_list std2(src_r)];
        end
        mean_list(i,k) = mean(std_list)


    end
    hold off
    %}

end
