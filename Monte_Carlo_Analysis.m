%% Data-Driven Stabilization Using Prior Knowledge on Stabilizability and Controllability
% Monte Carlo Analysis
% Date: October 20, 2025
% By: Tren M.J.T. Baltussen - Eindhoven University of Technology
% and Amir Shakouri - University of Groningen
% Contact: t.m.j.t.baltussen@tue.nl
%
% See the LICENSE file in the project root for full license information.

% This script uses YALMIP and SeDuMi.

clear
close all
clc

%% System Description - Three Tank System
% Define tank areas and flow coefficients
a1 = 1; a2 = 1; a3 = 1;
k01 = 0.1; k12 = 0.5; k23 = 0.5;
x0 = [0; 0; 0];     % Initial states

% Continuous-time state-space model
A_c = [-(k01+k12)/a1,    k12/a1,     0;
        k12/a2,          -k12/a2,    k23/a2;
        0,               0,          -k23/a3];
B_c = [0; 1/a2; 0];
C_c = [1, 0, 0];
D_c = 0;

% Create continuous-time and discretized systems
sys_c = ss(A_c, B_c, C_c, D_c);
Ts = 0.1;                       % Sampling time
sys_d = c2d(sys_c, Ts);         % Discretization
A = sys_d.A; B = sys_d.B; C = sys_d.C; D = sys_d.D;

n = size(A,1);                  % State dimension
m = size(B,2);                  % Input dimension

%% Monte Carlo Simulation for Data Informativity Analysis
T = 50;                            % Number of time steps per trial
N = 100;                           % Number of Monte Carlo runs - in paper: 500.

% Initialize result arrays
check_id     = zeros(1,N);         % System identification informativity
check_ddc    = zeros(1,N);         % Data-driven control informativity
check_ddc_pk = zeros(1,N);         % Stabilization informativity (persistent kernel)

for j = 1:N
    % Generate random initial condition and input sequence
    x0 = poissrnd(1,n,1);          % Random positive integers for initial states
    Um = poissrnd(1,m,T);          % Random non-negative excitation input
    % Alternative options:
    % x0 = randn(n,1);             % Gaussian initial states
    % Um = randn(m,T);             % Gaussian input signal

    % Generate data trajectory (X-, X+)
    [Xm, Xp] = data_gen(A, B, Um, x0);

    % Evaluate informativity for identification and control
    [check_id(j), check_ddc(j), check_ddc_pk(j)] = data_analysis(Um, Xm, Xp);
end

%% Display Statistical Results
disp(strcat('Informative for stabilization: ', num2str(sum(check_ddc_pk)/N*100), '%.'))
disp(strcat('Informative for control: ', num2str(sum(check_ddc)/N*100), '%.'))
disp(strcat('Informative for system identification: ', num2str(sum(check_id)/N*100), '%.'))

%% Visualization of Informative Ratios
figure;
clf
plot(cumsum(check_id)./cumsum(ones(size(check_id))), '-', 'LineWidth', 1.5); hold on;
plot(cumsum(check_ddc)./cumsum(ones(size(check_id))), '--', 'LineWidth', 1.5);
plot(cumsum(check_ddc_pk)./cumsum(ones(size(check_id))), '-', 'LineWidth', 1.5);
ylim([0 1]);
xlabel('Simulation Index', 'Interpreter','latex');
ylabel('Informative Ratio', 'Interpreter','latex');
legend({'Identification', 'Control', 'Stabilization'}, 'Interpreter','latex');
grid on;