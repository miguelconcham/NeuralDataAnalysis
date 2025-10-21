function [lags, meanCorr, ciLow, ciHigh] = lagged_window_corr_masked(X, Y, n, M, nWin, alpha, A)
% lagged_window_corr_masked computes lagged correlations from random windows
% restricted to samples where A == 1.
%
% Inputs:
%   X, Y   : signals (same length)
%   n      : maximum lag (compute -n:n)
%   M      : window size (samples)
%   nWin   : number of random windows
%   alpha  : significance level for CI (e.g., 0.05 for 95% CI)
%   A      : logical array (1 = valid, 0 = invalid), same length as X
%
% Outputs:
%   lags     : array of lags (-n:n)
%   meanCorr : mean correlation at each lag
%   ciLow    : lower CI
%   ciHigh   : upper CI

    if length(X) ~= length(Y) || length(X) ~= length(A)
        error('X, Y, and A must have the same length.');
    end
    N = length(X);
    lags = -n:n;
    numLags = length(lags);

    % --- find valid window start positions ---
    validStarts = find(A(1:end-M+1));  % potential start points
    keep = false(size(validStarts));

    for i = 1:length(validStarts)
        idx = validStarts(i):(validStarts(i)+M-1);
        if all(A(idx))   % only keep if the full window is within valid mask
            keep(i) = true;
        end
    end

    validStarts = validStarts(keep);

    if isempty(validStarts)
        error('No valid windows of length M found within A.');
    end

    % --- sample start indices ---
    if length(validStarts) < nWin
        warning('Not enough valid windows (%d), using all.', length(validStarts));
        starts = validStarts;
        nWin = length(validStarts);
    else
        starts = randsample(validStarts, nWin);
    end

    % --- compute correlations ---
    corrMat = nan(nWin, numLags);

    for w = 1:nWin
        if mod(w,10) == 0
            fprintf('Processing window %d of %d...\n', w, nWin);
        end

        idx = starts(w):(starts(w)+M-1);
        xw = X(idx);
        yw = Y(idx);

        for li = 1:numLags
            lag = lags(li);
            if lag >= 0
                xLag = xw(1:end-lag);
                yLag = yw(1+lag:end);
            else
                xLag = xw(1-lag:end);
                yLag = yw(1:end+lag);
            end

            if length(xLag) > 1
                r = corr(xLag(:), yLag(:), 'Rows','complete');
                corrMat(w,li) = r;
            end
        end
    end

    % --- summarize ---
    meanCorr = nanmean(corrMat,1);
    ciLow  = quantile(corrMat, alpha/2, 1);
    ciHigh = quantile(corrMat, 1-alpha/2, 1);
end
