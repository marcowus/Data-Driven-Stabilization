function [check, S, Xm_hat, Xp_hat] = ddlmi_pk(Um, Xm, Xp)
%% DDLMI_PK
% Tests data informativity for *stabilization* via the persistent-kernel
% approach. This method extends the data-driven LMI condition to cases
% where the measured data are rank-deficient, projecting them into a
% reduced-order subspace.
%
% INPUTS:
%   Um : Input data matrix (m x T)
%   Xm : State data matrix at time k     (n x T)
%   Xp : State data matrix at time k+1   (n x T)
%
% OUTPUTS:
%   check   : 1 if data are informative for stabilization, 0 otherwise
%   S       : State transformation matrix (n x n)
%   Xm_hat  : Projected state data matrix (r x T)
%   Xp_hat  : Projected next-state matrix (r x T)
%
% Date: October 20, 2025
% By: Tren M.J.T. Baltussen - Eindhoven University of Technology
%     Amir Shakouri - University of Groningen
% Contact: t.m.j.t.baltussen@tue.nl
%
% See the LICENSE file in the project root for full license information.

%% Dimensions and rank analysis
[n, T] = size(Xm);
[m, ~] = size(Um);
r = rank(Xm);                     % Numerical rank of state data

%% Case 1: Rank-deficient data (projection-based test)
if r < n
    [U, D, V] = svd(Xm);          % Singular value decomposition
    S = U';                       % Transformation matrix

    Xm_hat = D * V';              % Reduced-order state data
    Xp_hat = [eye(r), zeros(r, n-r)] * S * Xp;  % Projected next-state data

    r_xu = rank([Xm_hat; Um]);    % Combined rank of states and inputs
    kerXm = null(Xm');            % Kernel of Xm

    % Check persistent-kernel conditions
    if norm(kerXm' * Xp) == 0 && r_xu == r + m
        check = 1;                % Data informative for stabilization
    else
        check = 0;                % Not informative
    end

%% Case 2: Full-rank data (standard LMI test)
elseif r == n
    [check, ~] = ddlmi(Xm, Xp);   % Use standard DDLMI feasibility test

    % Fill outputs for consistency
    S = zeros(n, n);
    Xm_hat = zeros(n, T);
    Xp_hat = zeros(n, T);
end

end
