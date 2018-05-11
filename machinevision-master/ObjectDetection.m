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
lbp_list = zeros(4, length(D), 59);

% loop over all the images in the folder
for k = 1:length(D)
    src = imread(D(k).name);
    row = size(src,1);
    col = size(src,2);
    %center_point = [row/2 col/2]
    
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
         center = stats(list(i)).Centroid;
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
        %centers = [a(1) a(2)]
        plot(a(1),a(2),'b*');
        rectangle('Position',stats(list(i)).BoundingBox,'LineWidth',2,'LineStyle','--','EdgeColor','r');
        
                
        % Get extract lbp features in each window
        J = imcrop(src1, stats(list(i)).BoundingBox);
        lbp = extractLBPFeatures(J); 
        lbp_list(i, k, :) = lbp(:);

        % get five target windows around the centroids
        ax = round(a(1)-75);
        ay = round(a(2)-75);
        %rectangle('Position', [ax ay 151 151], 'LineWidth',1,'LineStyle','-','EdgeColor','r');
        aax = ax;
        aay = ay - 251;
        rectangle('Position', [aax aay 151 151], 'LineWidth',1,'LineStyle','-','EdgeColor','r');
        abx = ax;
        aby = ay + 251;
        rectangle('Position', [abx aby 151 151], 'LineWidth',1,'LineStyle','-','EdgeColor','r');
        alx = ax - 251;
        aly = ay;
        rectangle('Position', [alx aly 151 151], 'LineWidth',1,'LineStyle','-','EdgeColor','r');
        arx = ax + 251;
        ary = ay;
        rectangle('Position', [arx ary 151 151], 'LineWidth',1,'LineStyle','-','EdgeColor','r');
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

        %{
        if max(ax+151,col)==col & min(ax,1)==1 & max(ay+151,row)==row & min(ay,1)==1
            color = src((ay:ay+151),(ax:(ax+151)),:);
            R_m = color(:, :, 1);
            G_m = color(:, :, 2);
            B_m = color(:, :, 3);
            mean_red(5) = mean2(R_m);
            mean_green(5) = mean2(G_m);
            mean_blue(5) = mean2(B_m);
        end
        %}

        if max(aax+151,col)==col & min(aax,1)==1 & max(aay+151,row)==row & min(aay,1)==1
            color = src((aay:(aay+151)),(aax:(aax+151)),:);
            R_a = color(:, :, 1);
            G_a = color(:, :, 2);
            B_a = color(:, :, 3);
            mean_red(1) = mean2(R_a);
            mean_green(1) = mean2(G_a);
            mean_blue(1) = mean2(B_a);
        end

        if max(abx+151,col)==col & min(abx,1)==1 & max(aby+151,row)==row & min(aby,1)==1
            color = src((aby:(aby+151)),(abx:(abx+151)),:);
            R_b = color(:, :, 1);
            G_b = color(:, :, 2);
            B_b = color(:, :, 3);
            mean_red(2) = mean2(R_b);
            mean_green(2) = mean2(G_b);
            mean_blue(2) = mean2(B_b);
        end

        if max(alx+151,col)==col & min(alx,1)==1 & max(aly+151,row)==row & min(aly,1)==1
            color  = src((aly:(aly+151)),(alx:(alx+151)),:);
            R_l = color(:, :, 1);
            G_l = color(:, :, 2);
            B_l = color(:, :, 3);
            mean_red(3) = mean2(R_l);
            mean_green(3) = mean2(G_l);
            mean_blue(3) = mean2(B_l);

        end

        if max(arx+151,col)==col & min(arx,1)==1 & max(ary+151,row)==row & min(ary,1)==1
            color = src((ary:(ary+151)),(arx:(arx+151)),:);
            R_r = color(:, :, 1);
            G_r = color(:, :, 2);
            B_r = color(:, :, 3);
            mean_red(4) = mean2(R_r);
            mean_green(4) = mean2(G_r);
            mean_blue(4) = mean2(B_r);
        end

        mean_r = (mean_red(1)+mean_red(2)+mean_red(3)+mean_red(4))/5;
        mean_g = (mean_green(1)+mean_green(2)+mean_green(3)+mean_green(4))/5;
        mean_b = (mean_blue(1)+mean_blue(2)+mean_blue(3)+mean_blue(4))/5;

        R = [R_a R_b R_l R_r];
        G = [G_a G_b G_l G_r];
        B = [B_a B_b B_l B_r];

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
    % End Get entropy for each apple slice

    % Get contrast, correlation, energy for each apple slice
    format short g
    if stats(list(1)).Area > 400000
        e1 = round(entropy(src1(idx1)),4);
        glcm_pro1 = graycoprops(src(idx1));
        contrast1 = round(glcm_pro1.Contrast,4);
        correlation1 = round(glcm_pro1.Correlation*(10^17),4);
        energy1 = round(glcm_pro1.Energy*(10^3),4);
    else
        e1 = NaN;
        contrast1 = NaN;
        correlation1 = NaN;
        energy1 = NaN;
    end
    
    if stats(list(2)).Area > 400000
        e2 = round(entropy(src1(idx2)),4);
        glcm_pro2 = graycoprops(src(idx2));
        contrast2 = round(glcm_pro2.Contrast,4);
        correlation2 = round(glcm_pro2.Correlation*(10^16),4);
        energy2 = round(glcm_pro2.Energy*(10^3),4);
    else
        e2 = NaN;
        contrast2 = NaN;
        correlation2 = NaN;
        energy2 = NaN;
    end
    
    if stats(list(3)).Area > 400000
        e3 = round(entropy(src1(idx3)),4);
        glcm_pro3 = graycoprops(src(idx3));
        contrast3 = round(glcm_pro3.Contrast,4);
        correlation3 = round(glcm_pro3.Correlation*(10^16),4);
        energy3 = round(glcm_pro3.Energy*(10^3),4);
    else
        e3 = NaN;
        contrast3 = NaN;
        correlation3 = NaN;
        energy3 = NaN;
    end
    
    if stats(list(4)).Area > 400000
        e4 = round(entropy(src1(idx4)),4);
        glcm_pro4 = graycoprops(src(idx4));
        contrast4 = round(glcm_pro4.Contrast,4);
        correlation4 = round(glcm_pro4.Correlation*(10^16),4);
        energy4 = round(glcm_pro4.Energy*(10^3),4);
    else
        e4 = NaN;
        contrast4 = NaN;
        correlation4 = NaN;
        energy4 = NaN;
    end
    
    en_list(1,k) = e1;
    en_list(2,k) = e2;
    en_list(3,k) = e3;
    en_list(4,k) = e4;
    %en_list
    
    con_list(1,k) = contrast1;
    con_list(2,k) = contrast2;
    con_list(3,k) = contrast3;
    con_list(4,k) = contrast4;
    %con_list
    
    corr_list(1,k) = correlation1;
    corr_list(2,k) = correlation2;
    corr_list(3,k) = correlation3;
    corr_list(4,k) = correlation4;
    %corr_list
    
    ener_list(1,k) = energy1;
    ener_list(2,k) = energy2;
    ener_list(3,k) = energy3;
    ener_list(4,k) = energy4;
    %ener_list
    % End Get contrast, correlation, energy for each apple slice
    
    % End of getting texture characteristic
        
    hold off
    %}

end
%mean_list  



% Transfer NaN to mean value
n1 = nanmean(en_list);
nn1 = isnan(en_list);
ii1 = sum(nn1) < 4;
z1 = en_list(:,ii1);
z1(nn1(:,ii1)) = nonzeros(bsxfun(@times, nn1(:,ii1), n1(ii1)));
en_list(:,ii1) = z1;

n2 = nanmean(con_list);
nn2 = isnan(con_list);
ii2 = sum(nn2) < 4;
z2 = con_list(:,ii2);
z2(nn2(:,ii2)) = nonzeros(bsxfun(@times, nn2(:,ii2), n2(ii2)));
con_list(:,ii2) = z2;

n3 = nanmean(corr_list);
nn3 = isnan(corr_list);
ii3 = sum(nn3) < 4;
z3 = corr_list(:,ii3);
z3(nn3(:,ii3)) = nonzeros(bsxfun(@times, nn3(:,ii3), n3(ii3)));
corr_list(:,ii3) = z3;

n4 = nanmean(ener_list);
nn4 = isnan(ener_list);
ii4 = sum(nn4) < 4;
z4 = ener_list(:,ii4);
z4(nn4(:,ii4)) = nonzeros(bsxfun(@times, nn4(:,ii4), n4(ii4)));
ener_list(:,ii4) = z4;
% End of Transfer NaN to mean value

%all = zeros(length(D),15);
temp_en = [];
temp_con = [];
temp_ener = [];
temp_corr = [];
temp_area = [];
temp_circularity = [];
temp_cornerpoints = [];
temp_mean_R = [];
temp_mean_G = [];
temp_mean_B = [];
temp_sd_R = [];
temp_sd_G = [];
temp_sd_B = [];

for i = 1:length(D)
    for j = 1:4
        temp_en = [temp_en en_list(j,i)];
        temp_con = [temp_con con_list(j,i)];
        temp_ener = [temp_ener ener_list(j,i)];
        temp_corr = [temp_corr corr_list(j,i)];
        temp_area = [temp_area area_list(j,i)];
        temp_circularity = [temp_circularity circularity_list(j,i)];
        temp_cornerpoints = [temp_cornerpoints cornerpoints_list(j,i)];
        temp_mean_R = [temp_mean_R mean_R_list(j,i)];
        temp_mean_G = [temp_mean_G mean_G_list(j,i)];
        temp_mean_B = [temp_mean_B mean_B_list(j,i)];
        temp_sd_R = [temp_sd_R sd_R_list(j,i)];
        temp_sd_G = [temp_sd_G sd_G_list(j,i)];
        temp_sd_B = [temp_sd_B sd_B_list(j,i)];
    end
end

temp = reshape(lbp_list, 1, 4*length(D), 59);

temp_lbp = zeros(59, 4*length(D));
for s = 1:4*length(D)
    for p = 1:59
        temp_lbp(p, 1:4*length(D)) = temp(1, 1:4*length(D), p);
    end
end

all = horzcat(temp_en',temp_con',temp_ener',temp_corr',temp_area',temp_circularity',temp_cornerpoints',temp_mean_R',temp_mean_G',temp_mean_B',temp_sd_R',temp_sd_G',temp_sd_B', temp_lbp')

%{
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
%}
