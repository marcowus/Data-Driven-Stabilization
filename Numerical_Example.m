%% Data-Driven Stabilization Using Prior Knowledge on Stabilizability and Controllability
% Numerical Example
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
T = 3;                          % Number of data samples

%% Data Generation
% Generate random input excitation (persistently exciting for identification)
rng(1);
U = 5*randn(m,T);

% Simulate system to collect state data
X = zeros(3, T+1);
X(:,1) = x0;
for k = 1:T
    X(:,k+1) = A * X(:,k) + B * U(k);
end

X_  = X(:,1:end-1);             % Past states
X_p = X(:,2:end);               % Next states
r = rank(X_);                   % Effective state rank

%% Data-Driven Controller Computation
% Transformation matrix - In this example I satisfies the condition.
S = eye(n,n);

% In general: use SVD.
% [Q,D,V] = svd(X_); 
% S=Q';

% State projections (retain rank-r components)
SX_ = S*X_;    X_hat = SX_(1:r,:);
SX_p = S*X_p;  X_phat = SX_p(1:r,:);
U_ = U;

% Define and solve optimization problem using YALMIP
clear('yalmip')
Theta_var = sdpvar(T,r);

% Constraints ensure feasibility and consistency of data-driven model
F = [ X_hat * Theta_var == Theta_var' * X_hat', ...
      [X_hat * Theta_var, X_phat * Theta_var;
       Theta_var' * X_phat', X_hat * Theta_var] >= eye(2*r)*(1e-1) ];

Sol = sdpsettings('solver', 'sedumi');
optimize(F, [], Sol);

Theta = value(Theta_var);

% Controller gain extraction
K1 = ((X_hat * Theta)' \ (U_ * Theta)')';
K2 = 0*eye(n-r,n-r);
K  = [K1, K2]*S;

%% Closed-Loop Simulation
T_sim = 500;
X_sim = zeros(3, T_sim+1);
X_sim(:,1) = [-5, -5, 2];        % Nonzero initial condition

for k = 1:T_sim
    U_sim(k) = K * X_sim(:,k);   % State feedback control
    X_sim(:,k+1) = A * X_sim(:,k) + B * U_sim(k);
end

%% Visualization

figure(1)
clf
plot((0:T_sim)*Ts, X_sim(1,:), '-', 'LineWidth', 2); hold on;
plot((0:T_sim)*Ts, X_sim(2,:), '--', 'LineWidth', 2);
plot((0:T_sim)*Ts, X_sim(3,:), '-.', 'LineWidth', 2);
plot((0:(T_sim-1))*Ts, U_sim, ':', 'LineWidth', 2);
xlabel('Time [s]', 'Interpreter','latex');
ylabel('Value [-]', 'Interpreter','latex');
grid on;
legend('$x_1$','$x_2$','$x_3$','$u$', 'Interpreter','latex');
ylim([-5 5]); xlim([0 20]);
