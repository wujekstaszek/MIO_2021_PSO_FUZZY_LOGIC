warning off;
clear; clc;
close all;

%Wczytujemy dane
%Tworzymy fuzzy logic system
%Tworzymy zasady
%Puszczamy optymalizację za pomocą PSO(optymalizacja wag zasad i parametrów funkcji wejściowej)
global rules
global mf_inp
global inputs
global fis
global test
global learn
load seeds_dataset.txt
data = seeds_dataset;
data_size = size(data);
data=data(randperm(size(data, 1)), :);
inputs=size(data,2)-1;
test_size = data_size(1)/10;
mf_inp=3;
rules = mf_inp^inputs*mf_inp;
params = mf_inp * inputs * 3;
total_dim = rules + params;
mf_out = 3;



lb=zeros(total_dim);
ub=ones(total_dim);

input_names = {'input1'; 'input2';'input3'; 'input4';'input5';'input6';'input7';};
output_name = 'Seed class';

mins = min(data(:,1:7));
maxs = max(data(:,1:7));

data = [(data(:,1:7)-mins)./(maxs-mins),data(:,8)];

test=data(1:test_size,:);
learn=data(test_size+1:end,:);

mf_input_names = {'Bad','Medium','Good'};
mf_output_names = {'Class 1','Class 2','Class 3'};

fis = mamfis("NumInputs",inputs,"NumOutputs",1);
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
options = optimoptions('particleswarm','MaxIterations',500,'SwarmSize',250,'Display','iter','MaxStallIterations', 20, 'ObjectiveLimit', 0,"SelfAdjustmentWeight",4,"SocialAdjustmentWeight",4);
data_result = particleswarm(fun,total_dim,lb,ub,options);
test_function(data_result);
save("seedsc4c4","data_result");

%%

function result = test_function(vars)
    global fis
    global test
    updateVariables(vars)
    result = floor(evalfis(fis,test(:,1:7))*3+1);
    result = result == test(:,8);
end
function procentage_result = updateVariables(vars)
    global fis
    global learn
    global ruleList
    global rules
    global mf_inp
    global inputs
    global results
    max1 = rules;
    ruleList(:,9) = vars(1:max1);
    fis.Rules = [];
    fis = addRule(fis,ruleList);
    for i = 0:(inputs-1)
        for j = 0:(mf_inp-1)
            x=max1+i*9+j*3;
            temp = [vars(x+1),vars(x+2),vars(x+3)];
            fis.inputs(i+1).membershipfunctions(j+1).parameters = [min(temp),median(temp),max(temp)];
        end
    end
    results = floor(evalfis(fis,learn(:,1:7))*3+1);
    results = results ~= learn(:,8);
    
    
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



