function [check, Theta] = ddlmi(Xm, Xp)
%% DDLMI
% Evaluates data-driven control (DDC) informativity using a Linear Matrix
% Inequality (LMI) feasibility test based on collected state data.
%
% The method checks whether the dataset (Xm, Xp) is informative for the
% existence of a stabilizing feedback gain without explicitly identifying
% system matrices.
%
% INPUTS:
%   Xm : State data matrix at time k     (n x T)
%   Xp : State data matrix at time k+1   (n x T)
%
% OUTPUTS:
%   check : 1 if dataset is informative for DDC (LMI feasible), 0 otherwise
%   Theta : Feasible decision variable from the LMI (T x n)
%
% Date: October 20, 2025
% By: Tren M.J.T. Baltussen - Eindhoven University of Technology
%     Amir Shakouri - University of Groningen
% Contact: t.m.j.t.baltussen@tue.nl
%
% See the LICENSE file in the project root for full license information.

%% Problem setup
[n, T] = size(Xm);
Theta = sdpvar(T, n);              % Decision variable (YALMIP)

% LMI conditions enforcing data consistency and positive definiteness
C1 = [Xm*Theta,  Xp*Theta;
      Theta'*Xp', Xm*Theta];
C2 = Xm*Theta - (Xm*Theta)';       % Enforce symmetry constraint

Constraints = [C1 >= eye(2*n), C2 == 0];

options = sdpsettings('solver', 'sedumi', 'verbose', 0);

%% Solve feasibility problem
sol = optimize(Constraints, [], options);

%% Check feasibility and rank condition
if sol.problem == 0 && rank(Xm) == n
    check = 1;                     % Data informative for control
    Theta = value(Theta);
else
    check = 0;                     % Not informative or LMI infeasible
    Theta = zeros(T, n);
end

end
