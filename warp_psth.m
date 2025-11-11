function [psth_warped, t_warped] = warp_psth(psth, t, lengths, nbins, t_window)
% WARP_PSTH warps each row of PSTH according to event length.
%
% Inputs:
%   psth     - [n x T] matrix of responses
%   t        - [1 x T] time vector
%   lengths  - [n x 1] array of event durations
%   nbins    - scalar number of bins per phase
%   t_window - [1 x 2] array [t1 t2]
%
% Outputs:
%   psth_warped - [n x (3*nbins)] warped psth
%   t_warped    - [1 x (3*nbins)] warped time vector (for reference)

n = size(psth,1);
T = size(psth,2);

t1 = t_window(1);
t2 = t_window(2);

% Define warped time axis (normalized)
t_warped = linspace(t1, t2 + 5, 3*nbins); % just a placeholder visualization axis

% Preallocate
psth_warped = nan(n, 3*nbins);

for i = 1:n
    L = lengths(i);

    % --- Define 3 time regions for trial i ---
    mask1 = (t >= t1) & (t <= 0);
    mask2 = (t >= 0) & (t <= L);
    mask3 = (t >= L) & (t <= L + t2);

    % --- Original time segments ---
    tseg1 = t(mask1);
    tseg2 = t(mask2);
    tseg3 = t(mask3);

    yseg1 = psth(i, mask1);
    yseg2 = psth(i, mask2);
    yseg3 = psth(i, mask3);

  

    % --- Interpolate each segment ---
    y1w = interp1(tseg1, yseg1, linspace(t1, 0, nbins), 'linear', 'extrap');
    y2w = interp1(tseg2, yseg2, linspace(0, L, nbins), 'linear', 'extrap');
    y3w = interp1(tseg3, yseg3, linspace(L, L + t2, nbins), 'linear', 'extrap');

    % --- Concatenate warped parts ---
    psth_warped(i,:) = [y1w y2w y3w];
end
end
