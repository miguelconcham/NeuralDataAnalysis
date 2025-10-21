function [lags, meanCorr, ciLow, ciHigh] = lagged_window_corr(X, Y, n, M, nWin, alpha)
% lagged_window_corr computes lagged correlations from random windows
%
% Inputs:
%   X, Y   : signals (must be same length)
%   n      : maximum lag (compute -n:n)
%   M      : window size (samples)
%   nWin   : number of random windows to use
%   alpha  : significance level for CI (e.g., 0.05 for 95% CI)
%
% Outputs:
%   lags     : array of lags (-n:n)
%   meanCorr : mean correlation at each lag
%   ciLow    : lower bound of confidence interval
%   ciHigh   : upper bound of confidence interval

    if length(X) ~= length(Y)
        error('X and Y must have the same length.');
    end

    N = length(X);
    lags = -n:n;
    numLags = length(lags);

    % Preallocate correlation matrix
    corrMat = nan(nWin, numLags);

    % Draw random start indices for windows
    maxStart = N - M + 1;
    starts = randi(maxStart, nWin, 1);

    for w = 1:nWin
        
            fprintf('Processing window %d of %d...\n', w, nWin);
       
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

    % Mean correlation across sampled windows
    meanCorr = nanmean(corrMat,1);

    % Confidence intervals across windows
    ciLow  = quantile(corrMat, alpha/2, 1);
    ciHigh = quantile(corrMat, 1-alpha/2, 1);
end
