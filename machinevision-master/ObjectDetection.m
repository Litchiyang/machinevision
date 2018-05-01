clear;
close all

% load all the images into dataset

D = dir(fullfile('Apple_test','*JPG'));
fprintf('loading %02d',length(D));

for i = 1:length(D)
    fprintf('\b\b%02d',i);
end
fprintf('\n');

% end of load images

%mean_list = zeros(4,length(D));  % create mean values for each apple slice
en_list = zeros(4,length(D));  % create entropy values for each apple slice
con_list = zeros(4,length(D));  % create contrast values for each apple slice
ener_list = zeros(4,length(D));  % create energy values for each apple slice
corr_list = zeros(4,length(D));  % create correlation values for each apple slice
area_list = zeros(4,length(D));  
circularity_list = zeros(4,length(D)); 
cornerpoints_list = zeros(4,length(D));
mean_R_list = zeros(4,length(D));
mean_G_list = zeros(4,length(D));
mean_B_list = zeros(4,length(D));
sd_R_list = zeros(4,length(D));
sd_G_list = zeros(4,length(D));
sd_B_list = zeros(4,length(D));

% loop over all the images in the folder
for k = 1:length(D)
    src = imread(D(k).name);
    row = size(src,1);
    col = size(src,2);
    
    src1 = rgb2gray(src); % convert the original image into black/white
    src2 = medfilt2(src1,[5 5]); %convolution with image with kernel size 5*5

    level = graythresh(src2); %calculating threshold
    bw = im2bw(src,level); %binarization image

    se = strel('square',2); %create a 2*2 square
    SE1 = strel('square',3);
    contour = edge(src2,'canny');
    for i=1:6
      bw2 = imdilate(contour,SE1);
      contour = bw2;
    end
    bw3 = imerode(bw2,SE1);
    C = corner(bw3);
    openbw = imopen(bw,se); %smooth the edge 

    %stats = regionprops(openbw,'basic'); %get Area,Centroid,BoundingBox of the connected image
    stats = regionprops(openbw,'Area','Centroid','BoundingBox','Perimeter');
    centroids = cat(1,stats.Centroid);
    x_list = [];
    y_list = [];
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
        y = stats(i).BoundingBox(1);
        x = stats(i).BoundingBox(2);
        if stats(i).Area==apple1 | stats(i).Area==apple2 |stats(i).Area==apple3 |stats(i).Area==apple4
            list = [list,i];
        end
    end

    imshow(openbw),title('centroids')
    hold on

    for i=1:size(list,2)
        %%%%%%%%%%Calculating perimeters of each apple slice%%%%%%%%%
         perimeter = stats(list(i)).Perimeter;
         center = stats(list(i)).Centroid
         A = stats(list(i)).Area;
         area_list(i,k) = A; %the area(the number of pixels
         box = stats(list(i)).BoundingBox;
         width = box(3);
         height = box(4);
         ycoordinate = box(1);
         true_y = ycoordinate*std(y_list)+mean(y_list);
         xcoordinate = box(2);
         true_x = xcoordinate*std(x_list)+mean(x_list);
         xrange = true_x+width;
         yrange = true_y+height;
         numberPoints = 0;
         for j=1:size(C) %go through corner points
             y = C(1);
             x = C(2);
             if x<=xrange | y>=yrange |x>=xcoordinate |y<=ycoordinate
                 numberPoints=numberPoints+1;
             end
         end
         cornerpoints_list(i,k) = numberPoints; %the number of corner points
         circularity_list(i,k) = (4*pi*A)/perimeter^2; %circularity
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        
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
        % End get five target windows around the centroids
        
        % Get color mean std for the five windows
        R_m = [];
        G_m = [];
        B_m = [];

        R_a = [];
        G_a = [];
        B_a = [];

        R_b = [];
        G_b = [];
        B_b = [];

        R_l = [];
        G_l = [];
        B_l = [];

        R_r = [];
        G_r = [];
        B_r = [];

        mean_red = zeros(5);
        mean_green = zeros(5);
        mean_blue = zeros(5);


        if max(ax,row)==row & min(ax-101,1)==1 & max(ay+101,col)==col & min(ay,1)==1
            color = src(((ax-101):ax), (ay:ay+101), :);
            R_m = color(:, :, 1);
            G_m = color(:, :, 2);
            B_m = color(:, :, 3);
            mean_red(5) = mean2(R_m);
            mean_green(5) = mean2(G_m);
            mean_blue(5) = mean2(B_m);
        end

        if max(aax,row)==row & min(aax-101,1)==1 & max(aay+101,col)==col & min(aay,1)==1
            color = src(((aax-101):aax), (aay:aay+101), :);
            R_a = color(:, :, 1);
            G_a = color(:, :, 2);
            B_a = color(:, :, 3);
            mean_red(1) = mean2(R_a);
            mean_green(1) = mean2(G_a);
            mean_blue(1) = mean2(B_a);
        end

        if max(abx,row)==row & min(abx-101,1)==1 & max(aby+101,col)==col & min(aby,1)==1
            color = src(((abx-101):abx), (aby:aby+101), :);
            R_b = color(:, :, 1);
            G_b = color(:, :, 2);
            B_b = color(:, :, 3);
            mean_red(2) = mean2(R_b);
            mean_green(2) = mean2(G_b);
            mean_blue(2) = mean2(B_b);
        end

        if max(alx,row)==row & min(alx-101,1)==1 & max(aly+101,col)==col & min(aly,1)==1
            color  = src(((alx-101):alx), (aly:aly+101), :);
            R_l = color(:, :, 1);
            G_l = color(:, :, 2);
            B_l = color(:, :, 3);
            mean_red(3) = mean2(R_l);
            mean_green(3) = mean2(G_l);
            mean_blue(3) = mean2(B_l);

        end

        if max(arx,row)==row & min(arx-101,1)==1 & max(ary+101,col)==col & min(ary,1)==1
            color = src(((arx-101):arx), (ary:ary+101), :);
            R_r = color(:, :, 1);
            G_r = color(:, :, 2);
            B_r = color(:, :, 3);
            mean_red(4) = mean2(R_r);
            mean_green(4) = mean2(G_r);
            mean_blue(4) = mean2(B_r);
        end

        mean_r = (mean_red(1)+mean_red(2)+mean_red(3)+mean_red(4)+mean_red(5))/5;
        mean_g = (mean_green(1)+mean_green(2)+mean_green(3)+mean_green(4)+mean_green(5))/5;
        mean_b = (mean_blue(1)+mean_blue(2)+mean_blue(3)+mean_blue(4)+mean_blue(5))/5;

        R = [R_m R_a R_b R_l R_r];
        G = [G_m G_a G_b G_l G_r];
        B = [B_m B_a B_b B_l B_r];

        std_red = std2(R);
        std_green = std2(G);
        std_blue = std2(B);

        %fprintf('mean of R, G, B: %f, %f, %f\n', mean_r, mean_g, mean_b);
        %fprintf('standard deviation of R, G, B: %f, %f, %f\n', std_red, std_green, std_blue);
        mean_R_list(i,k) = mean_r;
        mean_G_list(i,k) = mean_g;
        mean_B_list(i,k) = mean_b;
        sd_R_list(i,k) = std_red;
        sd_G_list(i,k) = std_green;
        sd_B_list(i,k) = std_blue;
        % End Get color mean std for the five windows

    end

    % Get texture characteristic

    % Get entropy for each apple slice
    s = regionprops(openbw,'PixelIdxList');
    idx1 = s(1).PixelIdxList;
    idx2 = s(2).PixelIdxList;
    idx3 = s(3).PixelIdxList;
    idx4 = s(4).PixelIdxList;
    e1 = entropy(src1(idx1));
    e2 = entropy(src1(idx2));
    e3 = entropy(src1(idx3));
    e4 = entropy(src1(idx4));
    en_list(1,k) = e1;
    en_list(2,k) = e2;
    en_list(3,k) = e3;
    en_list(4,k) = e4;
    %en_list
    % End Get entropy for each apple slice

    % Get contrast, correlation, energy for each apple slice
    format short g
    glcm_pro1 = graycoprops(src(idx1));
    contrast1 = glcm_pro1.Contrast;
    correlation1 = glcm_pro1.Correlation*(10^16);
    energy1 = glcm_pro1.Energy;

    glcm_pro2 = graycoprops(src(idx2));
    contrast2 = glcm_pro2.Contrast;
    correlation2 = glcm_pro2.Correlation*(10^16);
    energy2 = glcm_pro2.Energy;
    
    glcm_pro3 = graycoprops(src(idx3));
    contrast3 = glcm_pro3.Contrast;
    correlation3 = glcm_pro3.Correlation*(10^16);
    energy3 = glcm_pro3.Energy;
    
    glcm_pro4 = graycoprops(src(idx4));
    contrast4 = glcm_pro4.Contrast;
    correlation4 = glcm_pro4.Correlation*(10^16);
    energy4 = glcm_pro4.Energy;
    
    con_list(1,k) = round(contrast1,4);
    con_list(2,k) = round(contrast2,4);
    con_list(3,k) = round(contrast3,4);
    con_list(4,k) = round(contrast4,4);
    %con_list
    
    corr_list(1,k) = round(correlation1,4);
    corr_list(2,k) = round(correlation2,4);
    corr_list(3,k) = round(correlation3,4);
    corr_list(4,k) = round(correlation4,4);
    %corr_list
    
    ener_list(1,k) = round(energy1,4);
    ener_list(2,k) = round(energy2,4);
    ener_list(3,k) = round(energy3,4);
    ener_list(4,k) = round(energy4,4);
    %ener_list
    % End Get contrast, correlation, energy for each apple slice
    
    % End of getting texture characteristic
        
    hold off
    %}

end
%mean_list  
en_list  
con_list   
ener_list  
corr_list
area_list  
circularity_list
cornerpoints_list
mean_R_list
mean_G_list
mean_B_list
sd_R_list
sd_G_list
sd_B_list

% Transfer NaN to 0
en_list(isnan(en_list))=0;
con_list(isnan(con_list))=0;
corr_list(isnan(corr_list))=0;
ener_list(isnan(ener_list))=0;
% End of Transfer NaN to 0
