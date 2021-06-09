clear; clc;

load iris.dat
iris_size = size(iris);
iris=iris(randperm(size(iris, 1)), :);
test=iris(1:15,:)
max1 = max(test)
min1 = min(test)


% test=iris(1:15,:)

