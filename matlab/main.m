
%Wczytujemy dane
%Tworzymy fuzzy logic system
%Tworzymy zasady
%Puszczamy optymalizację za pomocą PSO(optymalizacja wag zasad i parametrów funkcji wejściowej)


load iris.dat
iris_size = size(iris);
global fis
global test

test=iris(1:15,:);
fis = mamfis("NumInputs",3,"NumOutputs",1);
fis.name = "Iris classification problem fuzzy system"; 
for i = 1:3
    fis.inputs(i).membershipfunctions(1).name = "Bad";
    fis.inputs(i).membershipfunctions(2).name = "Medium";
    fis.inputs(i).membershipfunctions(3).name = "Good";
end
%DO OGARNIĘCIA RULSY, Chce ci się tym zająć kuba?

global numOfParametersPSO;
numOfParametersPSO = size(fis.rule)*3;
numOfParametersPSO = numOfParametersPSO(2);
fun=@(x)updateVariables(x);
particleswarm(fun,2)


function procentage_result = updateVariables(vars)
    global fis
    global test
    for i = 1:27
        fis.rules(i).weight = vars(i);
    end
    for i =1:3
        fis.input(i).memebershipfunctions(1).parameters = [vars(27+i*3+1),vars(27+i*3+2),vars(27+i*3+3)];
        fis.input(i).memebershipfunctions(2).parameters = [vars(27+i*3+4),vars(27+i*3+5),vars(27+i*3+6)];
        fis.input(i).memebershipfunctions(3).parameters = [vars(27+i*3+7),vars(27+i*3+8),vars(27+i*3+9)];
    end
    
    results = evalfis(fis,test(:,1:4));
    results = results == test(:,1:5);
    
    
    procentage_result = mean(results);
end



