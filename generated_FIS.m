warning off; clear; clc; close all;

%% AUTORZY
% Piotr Matiaszewski, Aleksander Morgała, Jakub Perlak

%% IRIS dataset
% LOAD SHUFFLED SET 
load iris.dat;                              % load set
data = iris(randperm(size(iris, 1)),:);     % shuffle set using permutation

% GET PROBLEM'S SIZES AND DIMENSIONS
data_sizes = size(data);

inputs_num = size(data,2)-1;        % number of input variables

% NORMALIZE DATA

mins = min(data(:,1:inputs_num));
maxs = max(data(:,1:inputs_num));
data = [(data(:,1:inputs_num)-mins)./(maxs-mins),data(:,inputs_num+1)];

% GET TEST AND LEARN SETS
test_size = data_sizes(1) * 0.1;     % size of test set
test = data(1:test_size,:);

learn = data(test_size+1:end,:);

% GENERATE FIS
opt = genfisOptions('SubtractiveClustering');
fis_generated = genfis(learn(:,1:inputs_num),learn(:,inputs_num+1),opt);
result_fis_generated = evalfis(fis_generated,test(:,1:inputs_num));

% PROJECT RESULTS TO CLASSES
for i = 1:size(result_fis_generated, 1)
    if result_fis_generated(i) <= 5/3
        result_fis_generated(i) = 1;
    elseif result_fis_generated(i) >= 7/3
        result_fis_generated(i) = 3;
    else
        result_fis_generated(i) = 2;
    end
end

% CHECK CORRECTNESS
res = result_fis_generated == test(:,inputs_num+1);
GENFIS_iris_correctness = mean(res)





%% HABERMAN'S SURVIVAL dataset
% LOAD SHUFFLED SET 
load dataset_haberman_survival.txt;                                                     % load set
data = dataset_haberman_survival(randperm(size(dataset_haberman_survival, 1)),:);       % shuffle set using permutation

% GET PROBLEM'S SIZES AND DIMENSIONS
data_sizes = size(data);

inputs_num = size(data,2)-1;        % number of input variables

% NORMALIZE DATA

mins = min(data(:,1:inputs_num));
maxs = max(data(:,1:inputs_num));
data = [(data(:,1:inputs_num)-mins)./(maxs-mins),data(:,inputs_num+1)];

% GET TEST AND LEARN SETS
test_size = data_sizes(1) * 0.1;     % size of test set
test = data(1:test_size,:);

learn = data(test_size+1:end,:);

% GENERATE FIS
opt = genfisOptions('SubtractiveClustering');
fis_generated = genfis(learn(:,1:inputs_num),learn(:,inputs_num+1),opt);
result_fis_generated = evalfis(fis_generated,test(:,1:inputs_num));

% PROJECT RESULTS TO CLASSES
for i = 1:size(result_fis_generated, 1)
    if result_fis_generated(i) >= 1.75
        result_fis_generated(i) = 2;
    else
        result_fis_generated(i) = 1;
    end
end

% CHECK CORRECTNESS
res = result_fis_generated == test(:,inputs_num+1);
GENFIS_hab_surv_correctness = mean(res)
