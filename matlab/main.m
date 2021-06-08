clear; clc;

%Wczytujemy dane
%Tworzymy fuzzy logic system
%Tworzymy zasady
%Puszczamy optymalizację za pomocą PSO(optymalizacja wag zasad i parametrów funkcji wejściowej)

load iris.dat
iris_size = size(iris);
iris=iris(randperm(size(iris, 1)), :);
global fis
global test
global iter
iter = 0 ;
lb=zeros(243);
ub=ones(243);

input_names = {'sepal length'; 'sepal width'; 'petal length'; 'petal width'};
output_name = 'iris class';

test=iris(1:15,:);
fis = mamfis("NumInputs",4,"NumOutputs",1);
fis.name = "Iris classification problem fuzzy system"; 
for i = 1:4
    fis.inputs(i).membershipfunctions(1).name = "Bad";
    fis.inputs(i).membershipfunctions(2).name = "Medium";
    fis.inputs(i).membershipfunctions(3).name = "Good";
    fis.inputs(i).name = input_names{i};
end
fis.outputs(1).membershipfunctions(1).name = "Class 1";
fis.outputs(1).membershipfunctions(2).name = "Class 2";
fis.outputs(1).membershipfunctions(3).name = "Class 3";
fis.outputs(1).name = output_name;
%DO OGARNIĘCIA RULSY
global ruleList
ruleList = get_rule_list(4, 3);

fis.Rules = [];
fis = addRule(fis, ruleList);

numOfParametersPSO = 243+36;
fun=@(x)updateVariables(x);
particleswarm(fun,numOfParametersPSO,lb,ub)

%%
function procentage_result = updateVariables(vars)
    global fis
    global test
    global iter
    global ruleList
    iter= iter+1
    max1 = 243;
    ruleList(:,6) = vars(1:243);
    fis.Rules = [];
    fis = addRule(fis,ruleList);
    for i =0:3
        temp1 = [vars(max1+i*9+1),vars(max1+i*9+2),vars(max1+i*9+3)];
        temp2 = [vars(max1+i*9+4),vars(max1+i*9+5),vars(max1+i*9+6)];
        temp3 = [vars(max1+i*9+7),vars(max1+i*9+8),vars(max1+i*9+9)];
        fis.inputs(i+1).membershipfunctions(1).parameters = [min(temp1),median(temp1),max(temp1)];
        fis.inputs(i+1).membershipfunctions(2).parameters = [min(temp2),median(temp2),max(temp2)];
        fis.inputs(i+1).membershipfunctions(3).parameters = [min(temp3),median(temp3),max(temp3)];
    end
    global results
    results = evalfis(fis,test(:,1:4));
    results = results == test(:,5);
    
    
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
        
        num = strcat(num,'01');
        
        for j = 1:(number_of_inputs+3)
            m(i+1,j) = str2double(num(j)) + 1;
        end
    end
end


