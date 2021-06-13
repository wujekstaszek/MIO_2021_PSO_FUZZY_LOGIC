warning off;
clear; clc;
close all;

%Wczytujemy dane
%Tworzymy fuzzy logic system
%Tworzymy zasady
%Puszczamy optymalizację za pomocą PSO(optymalizacja wag zasad i parametrów funkcji wejściowej)

load iris.dat
data = iris;
data_size = size(data);
data=data(randperm(size(data, 1)), :);
inputs=size(data,2)-1;
test_size = data_size(1)/10;
mf_inp=3;
rules = mf_inp^inputs*mf_inp;
params = mf_inp * inputs * 3;
total_dim = rules + params;
mf_out = 3;


global fis
global test
global learn
iter = 0 ;
lb=zeros(total_dim);
ub=ones(total_dim);

input_names = {'sepal length'; 'sepal width'; 'petal length'; 'petal width'};
output_name = 'Iris class';

mins = min(data(:,1:4));
maxs = max(data(:,1:4));

data = [(data(:,1:4)-mins)./(maxs-mins),data(:,5)];

test=data(1:test_size,:);
learn=data(test_size+1:end,:);

mf_input_names = {'Bad','Medium','Good'};
mf_output_names = {'Class 1','Class 2','Class 3'};

fis = mamfis("NumInputs",4,"NumOutputs",1);
fis.name = "Iris classification problem fuzzy system"; 
for i = 1:inputs
    for j = 1:mf_inp
        fis.inputs(i).membershipfunctions(j).name = mf_input_names{j};
    end
    fis.inputs(i).name = input_names{i};
end
output_params = [-0.5 0 0.5;0.25 0.5 0.75;0.5 1 1.5];
for j = 1:mf_out
fis.outputs(1).membershipfunctions(j).name = mf_output_names{j};
fis.outputs(1).mf(j).params = output_params(j,:);
end
fis.outputs(1).name = output_name;
%DO OGARNIĘCIA RULSY
global ruleList
ruleList = get_rule_list(inputs, mf_inp);

fis.Rules = [];
fis = addRule(fis, ruleList);

fun=@(x)updateVariables(x);
options = optimoptions('particleswarm','MaxIterations',500,'SwarmSize',100,'Display','iter','MaxStallIterations', 20, 'ObjectiveLimit', 0,"SelfAdjustmentWeight",3,"SocialAdjustmentWeight",3);
data_result = particleswarm(fun,total_dim,lb,ub,options);
test_function(data_result,test)
%%

function result = test_function(vars,test)
    updateVariables(vars)
    result = floor(evalfis(fis,test(:,1:4))*3+1);
    result = result == test(:,5);
end
function procentage_result = updateVariables(vars)
    global fis
    global learn
    global ruleList
    max1 = 243;
    ruleList(:,6) = vars(1:243);
    fis.Rules = [];
    fis = addRule(fis,ruleList);
    for i = 0:3
        temp1 = [vars(max1+i*9+1),vars(max1+i*9+2),vars(max1+i*9+3)];
        temp2 = [vars(max1+i*9+4),vars(max1+i*9+5),vars(max1+i*9+6)];
        temp3 = [vars(max1+i*9+7),vars(max1+i*9+8),vars(max1+i*9+9)];
        fis.inputs(i+1).membershipfunctions(1).parameters = [min(temp1),median(temp1),max(temp1)];
        fis.inputs(i+1).membershipfunctions(2).parameters = [min(temp2),median(temp2),max(temp2)];
        fis.inputs(i+1).membershipfunctions(3).parameters = [min(temp3),median(temp3),max(temp3)];
    end
    global results
    results = floor(evalfis(fis,learn(:,1:4))*3+1);
    results = results ~= learn(:,5);
    
    
    procentage_result = mean(results);
end

function m = get_rule_list(number_of_inputs, number_of_rules_values)
    num_of_combinations = number_of_rules_values^(number_of_inputs+1);
    m = zeros(num_of_combinations,number_of_inputs+3,'double'); % 1 for output, 1 for weight, 1 for and/or

    for i = 0:num_of_combinations-1
        num = dec2base(i,number_of_rules_values);
        len_diff = number_of_inputs + 1 - strlength(num);
        
        if len_diff > 0
           zerros = '';
           for j = 1:len_diff
               zerros = strcat('0', zerros);
           end
           num = strcat(zerros,num);
        end
        
        num = strcat(num,'00');
        
        for j = 1:(number_of_inputs+3)
            m(i+1,j) = str2double(num(j)) + 1;
        end
    end
end



