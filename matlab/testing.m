clear; clc;

load iris.dat
iris_size = size(iris);
iris=iris(randperm(size(iris, 1)), :);
test=iris(1:15,:);
max1 = max(test(:,1:4))
min1 = min(test(:,1:4))
max1-min1
test
(test(:,1:4)-min1)./(max1-min1)

% test=iris(1:15,:)

