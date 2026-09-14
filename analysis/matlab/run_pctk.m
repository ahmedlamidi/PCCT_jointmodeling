function run_pctk(which)
%RUN_PCTK  Launch a PcTK workflow with the toolbox shims on the path.
%
%   run_pctk('nocorr')  uncorrelated noise, ~1 min   (script_workflow_PcTK_NoCorrelation)
%   run_pctk('corr')    correlated noise,   6-7 h    (script_workflow_PcTK)
%   run_pctk('gennc')   regenerate nCovE/nCovW, ~70 min at 225 um (gen_nCovE)
%
%   Needed because script_workflow_PcTK.m never adds ./3_src itself, and because
%   this machine has no Statistics and Machine Learning Toolbox -- ../matlab
%   supplies mvnrnd/normrnd/random/poissrnd in base MATLAB.
%
%   Both workflow scripts write the SAME output filenames. Rename the outputs of
%   one before running the other, or the second overwrites the first.

if nargin < 1, which = 'nocorr'; end
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(here));          % .../PCCT_jointmodeling
pctk = fullfile(root, 'PcTK_3.24a');

addpath(here);                               % toolbox shims
addpath(fullfile(pctk, '3_src'));
old  = cd(pctk);                             % scripts use relative paths
cleaner = onCleanup(@() cd(old));

t0 = tic;
switch lower(which)
    case 'nocorr', script_workflow_PcTK_NoCorrelation
    case 'corr',   script_workflow_PcTK
    case 'gennc',  gen_nCovE
    otherwise, error('run_pctk: unknown target ''%s''', which);
end
fprintf('RUN_DONE (%s) elapsed %.1f s\n', which, toc(t0));
end
