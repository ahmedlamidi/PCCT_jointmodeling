function r = mvnrnd(mu, Sigma, n)
%MVNRND  Base-MATLAB stand-in for the Statistics Toolbox function.
%   Falls back to an eigendecomposition when Sigma is not quite PSD.
if nargin < 3, n = 1; end
mu = mu(:).';  d = numel(mu);
S  = (Sigma + Sigma.') / 2;
[R, p] = chol(S);
if p > 0
    [V, D] = eig(S);
    R = (V * diag(sqrt(max(real(diag(D)), 0)))).';
end
r = repmat(mu, n, 1) + randn(n, d) * R;
end
