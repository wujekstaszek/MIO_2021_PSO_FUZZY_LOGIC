
%Wczytujemy dane
%Tworzymy fuzzy logic system
%Tworzymy zasady
%Puszczamy optymalizację za pomocą PSO(optymalizacja wag zasad i parametrów funkcji wejściowej)


load iris.dat
iris_size = size(iris);
iris=iris(randperm(size(iris, 1)), :);
global fis
global test
lb=zeros(100);
ub=ones(100);

test=iris(1:15,:);
fis = mamfis("NumInputs",4,"NumOutputs",1);
fis.name = "Iris classification problem fuzzy system"; 
for i = 1:4
    fis.inputs(i).membershipfunctions(1).name = "Bad";
    fis.inputs(i).membershipfunctions(2).name = "Medium";
    fis.inputs(i).membershipfunctions(3).name = "Good";
end
%DO OGARNIĘCIA RULSY


numOfParametersPSO = size(fis.rule)*3;
numOfParametersPSO = numOfParametersPSO(2);
fun=@(x)updateVariables(x);
particleswarm(fun,numOfParametersPSO,lb,ub)

%%
function procentage_result = updateVariables(vars)
    global fis
    global test
    for i = 1:36
        fis.rules(i).weight = vars(i);
    end
    for i =1:4
        temp1 = [vars(36+i*3+1),vars(36+i*3+2),vars(36+i*3+3)];
        temp2 = [vars(36+i*3+4),vars(36+i*3+5),vars(36+i*3+6)];
        temp3 = [vars(36+i*3+7),vars(36+i*3+8),vars(36+i*3+9)];
        fis.inputs(i).membershipfunctions(1).parameters = [min(temp1),median(temp1),max(temp1)];
        fis.inputs(i).membershipfunctions(2).parameters = [min(temp2),median(temp2),max(temp2)];
        fis.inputs(i).membershipfunctions(3).parameters = [min(temp3),median(temp3),max(temp3)];
    end
    global results
    results = evalfis(fis,test(:,1:4));
    results = results == test(:,5);
    
    
    procentage_result = mean(results);
end



