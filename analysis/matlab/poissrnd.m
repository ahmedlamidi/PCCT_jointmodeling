function r = poissrnd(lambda, varargin)
%POISSRND  Base-MATLAB stand-in. Knuth below 30, normal approximation above.
if nargin > 1, sz = [varargin{:}]; else, sz = size(lambda); end
lam = lambda .* ones(sz);  r = zeros(sz);
small = lam < 30 & lam >= 0;
idx = find(small);
for k = 1:numel(idx)
    L = exp(-lam(idx(k)));  kk = 0;  p = 1;
    while true
        p = p * rand;
        if p <= L, break; end
        kk = kk + 1;
        if kk > 1e6, break; end
    end
    r(idx(k)) = kk;
end
big = ~small & lam >= 0;
r(big) = max(0, round(lam(big) + sqrt(lam(big)) .* randn(sum(big(:)), 1)));
end
