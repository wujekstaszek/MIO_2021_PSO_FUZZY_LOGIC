warning off;
clear; clc;
close all;

%Wczytujemy dane
%Tworzymy fuzzy logic system
%Tworzymy zasady
%Puszczamy optymalizację za pomocą PSO(optymalizacja wag zasad i parametrów funkcji wejściowej)

load haberman_survival_dataset.txt
data = haberman_survival_dataset;
data_size = size(data);
data=data(randperm(size(data, 1)), :);
inputs=size(data,2)-1;
test_size = data_size(1)/10;
mf_inp=3;
rules = mf_inp^inputs*2;
params = mf_inp * inputs * 3;
total_dim = rules + params;
mf_out = 2;


global fis
global test
global learn
iter = 0 ;
lb=zeros(total_dim);
ub=ones(total_dim);

input_names = {'patient age'; 'operation year'; 'axillary nodes'};
output_name = 'Survival status';

mins = min(data(:,1:inputs));
maxs = max(data(:,1:inputs));

data = [(data(:,1:inputs)-mins)./(maxs-mins),data(:,inputs+1)];

test=data(1:test_size,:);
learn=data(test_size+1:end,:);

mf_input_names = {'Bad','Medium','Good'};
mf_output_names = {'Survived','Died'};

fis = mamfis("NumInputs",inputs,"NumOutputs",1);
fis.name = "Haberman survival problem fuzzy system"; 
for i = 1:inputs
    for j = 1:mf_inp
        fis.inputs(i).membershipfunctions(j).name = mf_input_names{j};
    end
    fis.inputs(i).name = input_names{i};
end
output_params = [-0.25 0.5 1;0.5 1 1];
for j = 1:mf_out
fis.outputs(1).membershipfunctions(j).name = mf_output_names{j};
fis.outputs(1).mf(j).params = output_params(j,:);
end
fis.outputs(1).name = output_name;
fis = removeMF(fis,output_name,'mf3','VariableType',"output");
%DO OGARNIĘCIA RULSY
global ruleList
ruleList = get_rule_list(inputs, mf_inp, mf_out);

fis.Rules = [];
fis = addRule(fis, ruleList);

fun=@(x)updateVariables(x);
options = optimoptions('particleswarm','MaxIterations',500,'SwarmSize',200,'Display','iter','MaxStallIterations', 50, 'ObjectiveLimit', 0,"SelfAdjustmentWeight",2,"SocialAdjustmentWeight",2);
data_result = particleswarm(fun,total_dim,lb,ub,options);
test_function(data_result);
%%

function result = test_function(vars)
    global fis
    global test
    updateVariables(vars);
    result = floor(evalfis(fis,test(:,1:3))*(4/3)+1);
    result = result == test(:,4);
end
function procentage_result = updateVariables(vars)
    global fis
    global learn
    global ruleList
    max1 = 54;
    ruleList(:,5) = vars(1:54);
    fis.Rules = [];
    fis = addRule(fis,ruleList);
    for i = 0:2
        temp1 = [vars(max1+i*9+1),vars(max1+i*9+2),vars(max1+i*9+3)];
        temp2 = [vars(max1+i*9+4),vars(max1+i*9+5),vars(max1+i*9+6)];
        temp3 = [vars(max1+i*9+7),vars(max1+i*9+8),vars(max1+i*9+9)];
        fis.inputs(i+1).membershipfunctions(1).parameters = [min(temp1),median(temp1),max(temp1)];
        fis.inputs(i+1).membershipfunctions(2).parameters = [min(temp2),median(temp2),max(temp2)];
        fis.inputs(i+1).membershipfunctions(3).parameters = [min(temp3),median(temp3),max(temp3)];
    end
    global results
    results = floor(evalfis(fis,learn(:,1:3))*(4/3)+1);
    results = results ~= learn(:,4);
    
    
    procentage_result = mean(results);
end

function m = get_rule_list(number_of_inputs, number_of_rules_values, num_of_output_classes)
    num_of_combinations = number_of_rules_values^number_of_inputs;
    m = zeros(num_of_combinations*num_of_output_classes,number_of_inputs+3,'double'); % 1 for output, 1 for weight, 1 for and/or
    
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
                m(1+2*i+class_ind,k) = str2double(num2(k)) + 1;
            end
        end
    end
end


