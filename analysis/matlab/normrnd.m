function r = normrnd(mu, sigma, varargin)
%NORMRND  Base-MATLAB stand-in for the Statistics Toolbox function.
if nargin > 2, sz = [varargin{:}]; else, sz = size(mu + sigma); end
r = mu + sigma .* randn(sz);
end
