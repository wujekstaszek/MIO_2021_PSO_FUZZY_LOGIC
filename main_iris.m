warning off; clear; clc; close all;

%% AUTHORS
% Piotr Matiaszewski, Aleksander Morgała, Jakub Perlak

%% DATA PREPERATION
% LOAD SHUFFLED SET 
load iris.dat;                              % load set
data = iris(randperm(size(iris, 1)),:);     % shuffle set using permutation

% GET PROBLEM'S SIZES AND DIMENSIONS
data_sizes = size(data);

global inputs_num
inputs_num = size(data,2)-1;        % number of input variables
outputs_num = size(data,2)-inputs_num;

mf_input_values = 3;                % number of values in input functions
global mf_output_classes_num
mf_output_classes_num = 3;          % number of classes in output functions

global rules_num
rules_num = mf_input_values^inputs_num * mf_output_classes_num;
params_num = mf_input_values * inputs_num * 3;
total_dim = rules_num + params_num;

% NORMALIZE DATA

mins = min(data(:,1:inputs_num));
maxs = max(data(:,1:inputs_num));
data = [(data(:,1:inputs_num)-mins)./(maxs-mins),data(:,inputs_num+1)];

% GET TEST AND LEARN SETS
global test
test_size = data_sizes(1) * 0.1;     % size of test set
test = data(1:test_size,:);

global learn
learn = data(test_size+1:end,:);

% FUZZY LOGIC DEFINITION
global fis
fis = mamfis("NumInputs",inputs_num,"NumOutputs",outputs_num);
fis.name = "Iris classification problem fuzzy system";                          % FIS name

% FL INPUTS
input_names = {'sepal length'; 'sepal width'; 'petal length'; 'petal width'};   % input variables' names
mf_input_names = {'Bad','Medium','Good'};                                       % names of input variables' membership functions

for i = 1:inputs_num
    for j = 1:mf_input_values
        fis.inputs(i).membershipfunctions(j).name = mf_input_names{j};
    end
    fis.inputs(i).name = input_names{i};
end

% FL OUTPUTS
output_names = ['Iris class'];                                                  % outputs' names
mf_output_names = {'Class 1','Class 2','Class 3'};                              % names of output variables' membership functions
output_params = [-0.5 0 0.5; 0.25 0.5 0.75; 0.5 1 1.5];                         % parameters of output variables' membership functions

for i = 1:outputs_num
    for j = 1:mf_output_classes_num
        fis.outputs(i).membershipfunctions(j).name = mf_output_names{j};
        fis.outputs(i).membershipfunctions(j).params = output_params(j,:);
    end
    fis.outputs(i).name = output_names(i);
end

%% RULES
global ruleList
ruleList = get_rule_list(inputs_num, mf_input_values, mf_output_classes_num);
fis.Rules = [];
fis = addRule(fis, ruleList);

%% PSO
fitness_function = @(x)updateVariables(x);

lb = zeros(total_dim);
ub = ones(total_dim);

options = optimoptions('particleswarm',...                                  % PSO options
                       'MaxIterations',500,...
                       'SwarmSize',100,...
                       'Display','iter',...
                       'MaxStallIterations',50,...
                       'ObjectiveLimit', 0,...
                       "SelfAdjustmentWeight",2,...
                       "SocialAdjustmentWeight",2);
data_result = particleswarm(fitness_function,total_dim,lb,ub,options);

%% TESTING
test_function(data_result)





%% FUNCTION TO TEST CORRECTNESS OF ALGORITHM
function result = test_function(vars)
    global fis
    global test
    global inputs_num
    updateVariables(vars);
    result = floor(evalfis(fis,test(:,1:inputs_num))*3+1);      % projecting results to classes
    result = result == test(:,inputs_num+1);
end

%% FITNESS FUNCTION
function procentage_result = updateVariables(vars)
    global fis
    global learn
    global ruleList
    global inputs_num
    global rules_num
    ruleList(:,inputs_num+2) = vars(1:rules_num);
    fis.Rules = [];
    fis = addRule(fis,ruleList);
    for i = 0:inputs_num-1
        temp1 = [vars(rules_num+i*9+1),vars(rules_num+i*9+2),vars(rules_num+i*9+3)];
        temp2 = [vars(rules_num+i*9+4),vars(rules_num+i*9+5),vars(rules_num+i*9+6)];
        temp3 = [vars(rules_num+i*9+7),vars(rules_num+i*9+8),vars(rules_num+i*9+9)];
        fis.inputs(i+1).membershipfunctions(1).parameters = [min(temp1),median(temp1),max(temp1)];
        fis.inputs(i+1).membershipfunctions(2).parameters = [min(temp2),median(temp2),max(temp2)];
        fis.inputs(i+1).membershipfunctions(3).parameters = [min(temp3),median(temp3),max(temp3)];
    end
    global results
    results = floor(evalfis(fis,learn(:,1:inputs_num))*3+1);    % projecting results to classes
    results = results ~= learn(:,inputs_num+1);
    
    procentage_result = mean(results);
end

%% FUNCTION TO GET ALL COMBINATION OF RULES
function m = get_rule_list(number_of_inputs, number_of_rules_values, num_of_output_classes)
    num_of_combinations = number_of_rules_values^number_of_inputs;
    m = zeros(num_of_combinations*num_of_output_classes,number_of_inputs+3,'double');
    
    for i = 0:num_of_combinations-1
        num = dec2base(i,number_of_rules_values);
        len_diff = number_of_inputs - strlength(num);
        
        if len_diff > 0
           zerros = '';
           for j = 1:len_diff
               zerros = strcat('0', zerros);
           end
           num = strcat(zerros,num);
        end
        
        for class_ind = 0:num_of_output_classes-1
            num2 = num;
            
            num2 = strcat(num2,int2str(class_ind));
        
            num2 = strcat(num2,'00');
            
            for k = 1:(number_of_inputs+3)
                m(1+num_of_output_classes*i+class_ind,k) = str2double(num2(k)) + 1;
            end
        end
    end
end
