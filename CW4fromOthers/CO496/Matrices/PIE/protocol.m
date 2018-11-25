clear; clc; close all

load Train5_64;
load fea64;
load gnd64;

fea = fea64; clear fea64;
gnd = gnd64; clear gnd64;
Train = Train5_64; clear Train5_64;

fea1 = fea;

%%% Train contains 20 random splits of the data
%%% gnd is the matrix with data labels (i.e., face identities)


error_PCA = [];
error_wPCA = [];
error_LDA = [];

dim_PCA = 10; %%check recognition rate every dim dimensions
dim_LDA = 5;

for jj = 1:20  %%%run for 20 random pertrurbations
    
    TrainIdx = Train(jj, :);
    TestIdx = 1:size(fea, 1);
    TestIdx(TrainIdx) = [];
    
    fea_Train = fea1(TrainIdx,:);
    gnd_Train = gnd(TrainIdx);
    [gnd_Train ind] = sort(gnd_Train, 'ascend');
    fea_Train = fea_Train(ind, :);
    
    fea_Test = fea1(TestIdx,:);
    gnd_Test = gnd(TestIdx);
    
    %%%%%%%%%%%%% PCA %%%%%%%%%%%%%
    U_reduc = PCA_SSS(fea_Train, 0);
    
    %%%dimensionality reduction
    oldfea = fea_Train*U_reduc;  %%of the training data
    newfea = fea_Test*U_reduc;   %%of the test data
    
    mg = mean(oldfea, 1);  %%compute the training mean
    mg_oldfea = repmat(mg,  size(oldfea,1), 1);
    oldfea = oldfea - mg_oldfea; %%subtract the mean
    
    mg_newfea = repmat(mg,  size(newfea,1), 1);
    newfea = newfea - mg_newfea;  %%subtract the mean
    
    len_PCA = 1:dim_PCA:size(newfea, 2);
    correct = zeros(1, length(1:dim_PCA:size(newfea, 2)));
    for ii = 1:length(len_PCA)  %%for each dimension perform classification
        Sample = newfea(:, 1:len_PCA(ii));
        Training = oldfea(:, 1:len_PCA(ii));
        Group = gnd_Train;
        k = 1;
        distance = 'cosine';
        Class = knnclassify(Sample, Training , Group, k, distance); %%nearest neighbor classification
        
        correct(ii) = length(find(Class-gnd_Test == 0));
    end
    
    correct = correct./length(gnd_Test); %%compute the correct classification rate
    error_PCA = [error_PCA; 1- correct];
    
    
    %%%%%%%%%%%%% PCA, with whitening %%%%%%%%%%%%%
    U_reduc = PCA_SSS(fea_Train, 1);
    
    %%%dimensionality reduction
    oldfea = fea_Train*U_reduc;  %%of the training data
    newfea = fea_Test*U_reduc;   %%of the test data
    
    mg = mean(oldfea, 1);  %%compute the training mean
    mg_oldfea = repmat(mg,  size(oldfea,1), 1);
    oldfea = oldfea - mg_oldfea; %%subtract the mean
    
    mg_newfea = repmat(mg,  size(newfea,1), 1);
    newfea = newfea - mg_newfea;  %%subtract the mean
    
    len_PCA = 1:dim_PCA:size(newfea, 2);
    correct = zeros(1, length(1:dim_PCA:size(newfea, 2)));
    for ii = 1:length(len_PCA)  %%for each dimension perform classification
        Sample = newfea(:, 1:len_PCA(ii));
        Training = oldfea(:, 1:len_PCA(ii));
        Group = gnd_Train;
        k = 1;
        distance = 'cosine';
        Class = knnclassify(Sample, Training , Group, k, distance); %%nearest neighbor classification
        
        correct(ii) = length(find(Class-gnd_Test == 0));
    end
    
    correct = correct./length(gnd_Test); %%compute the correct classification rate
    error_wPCA = [error_wPCA; 1- correct];
    
    
    %%%%%%%%%%%%% LDA %%%%%%%%%%%%%
    U_reduc = LDA_SSS(gnd_Train, fea_Train);
    
    %%%dimensionality reduction
    oldfea = fea_Train*U_reduc;  %%of the training data
    newfea = fea_Test*U_reduc;   %%of the test data
    
    mg = mean(oldfea, 1);  %%compute the training mean
    mg_oldfea = repmat(mg,  size(oldfea,1), 1);
    oldfea = oldfea - mg_oldfea; %%subtract the mean
    
    mg_newfea = repmat(mg,  size(newfea,1), 1);
    newfea = newfea - mg_newfea;  %%subtract the mean
    
    len_LDA = 1 : dim_LDA : size(newfea, 2);
    correct = zeros(1, length(1 : dim_LDA : size(newfea, 2)));
    for ii = 1:length(len_LDA)  %%for each dimension perform classification
        Sample = newfea(:, 1:len_LDA(ii));
        Training = oldfea(:, 1:len_LDA(ii));
        Group = gnd_Train;
        k = 1;
        distance = 'cosine';
        Class = knnclassify(Sample, Training , Group, k, distance); %%nearest neighbor classification
        
        correct(ii) = length(find(Class-gnd_Test == 0));
    end
    
    correct = correct./length(gnd_Test); %%compute the correct classification rate
    error_LDA = [error_LDA; 1- correct];
    
end

% Would obtain 20 errors per dimension (20 x dim matrix).
% Plot mean error over each dimension against number of dimensions.
figure;
plot(len_PCA, mean(error_PCA), len_PCA, mean(error_wPCA));
hold on;

plot(len_LDA, mean(error_LDA), 'k');
xlabel('Number of dimensions', 'fontsize', 14);
ylabel('Test errors', 'fontsize', 14);
legend = legend('PCA', 'PCA with whitening', 'LDA');
title('Test errors against number of dimensions for PIE dataset', 'fontsize', 14);

set(legend, 'fontsize', 14);