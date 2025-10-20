function [check_id, check_ddc, check_ddc_pk] = data_analysis(Um, Xm, Xp)
%% DATA_ANALYSIS
% Evaluates the informativeness of collected data for:
%   1. System identification
%   2. Data-driven control design
%   3. Stabilization via persistently informative kernels
%
% INPUTS:
%   Um : Input data matrix (m x T)
%   Xm : State data matrix at time k (n x T)
%   Xp : State data matrix at time k+1 (n x T)
%
% OUTPUTS:
%   check_id     : 1 if data are informative for identification, 0 otherwise
%   check_ddc    : 1 if data are informative for data-driven control
%   check_ddc_pk : 1 if data are informative for stabilization (persistent kernel)
%
% Date: September 18, 2025
% By: Tren M.J.T. Baltussen - Eindhoven University of Technology
%     Amir Shakouri - University of Groningen
% Contact: t.m.j.t.baltussen@tue.nl
%
% See the LICENSE file in the project root for full license information.

%% Dimensions
[n,~] = size(Xm);
[m,~] = size(Um);

%% Data-Driven Control Informativity Tests
% The ddlmi functions evaluate feasibility of LMIs linked to data-based
% stabilizability and control design.
[check_ddc, ~] = ddlmi(Xm, Xp);                   % Basic DDC informativity
[check_ddc_pk, ~, ~, ~] = ddlmi_pk(Um, Xm, Xp);   % Stabilization informativity

%% System Identification Informativity
% Data are informative for identification if [Xm; Um] has full row rank (n + m)
if rank([Xm; Um]) == n + m
    check_id = 1;
else
    check_id = 0;
end

end
