function r = random(name, varargin)
%RANDOM  Base-MATLAB stand-in covering the distributions PcTK uses.
switch lower(name)
    case {'poisson','poiss'},  r = poissrnd(varargin{:});
    case {'normal','norm'},    r = normrnd(varargin{:});
    otherwise, error('random: distribution ''%s'' not supported by this shim.', name);
end
end
