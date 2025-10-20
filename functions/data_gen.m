function [Xm, Xp] = data_gen(A, B, Um, x0)
%% DATA_GEN
% Generates state trajectory data for a discrete-time linear system:
%
%       x(k+1) = A*x(k) + B*u(k)
%
% using a given input sequence and initial state.
%
% INPUTS:
%   A  : State matrix (n x n)
%   B  : Input matrix (n x m)
%   Um : Input sequence matrix (m x T)
%   x0 : Initial state vector (n x 1)
%
% OUTPUTS:
%   Xm : State data matrix at time k     (n x T)
%   Xp : State data matrix at time k+1   (n x T)
%
% Date: October 20, 2025
% By: Tren M.J.T. Baltussen - Eindhoven University of Technology
%     Amir Shakouri - University of Groningen
% Contact: t.m.j.t.baltussen@tue.nl
%
% See the LICENSE file in the project root for full license information.

%% Initialize and simulate system
[~, T] = size(Um);      % Determine horizon length
x = zeros(size(A,1), T+1);
x(:,1) = x0;            % Set initial condition

% Propagate dynamics over T steps
for t = 1:T
    x(:,t+1) = A*x(:,t) + B*Um(:,t);
end

%% Collect data matrices
Xm = x(:,1:T);          % State matrix at time k
Xp = x(:,2:T+1);        % State matrix at time k+1

end
