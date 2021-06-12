clc;
clear;
close all;

%% Problem Definition

problem.CostFunction = @(x) Sphere(x);      % Cost Function
problem.nVar = 5;                           % Number of Unknown (Decision) Variables
problem.VarMin = -10;                       % Lower Bound of Decision Variables
problem.VarMax = 10;                        % Upper Bound of Decision Variables


%% Parameters of PSO

params.MaxIt = 1000;               % Maximum Number of Iterations
params.nPop = 50;                  % Population Size (Swarm Size)
params.w = 1;                      % Intertia Coefficient
params.wdamp = 0.99;               % Damping Ratio of Inertia Coefficient
params.c1 = 2;                     % Personal Acceleration Coefficient
params.c2 = 2;                     % Social Acceleration Coefficient
params.ShowIterationsInfo = true;  % Flag for Showing Itration Information


%% Calling PSO

out = PSO(problem, params);

BestSolution = out.BestSolution;
BestCosts = out.BestCost;


%% Results

figure;
% plot(BestCosts, 'LineWidth', 2);
semilogy(BestCosts, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;
