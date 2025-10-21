function results = testRepeatedOverTime(A, X, z)
% testRepeatedOverTime
%   Tests if X(:,t) is significantly greater than z in a repeated measures design.
%
%   INPUTS:
%     A : [N x 1] categorical vector of subject IDs
%     X : [N x T] numeric matrix (N subjects × T timepoints)
%     z : scalar threshold
%
%   OUTPUT:
%     results : table with columns
%       Time       - timepoint index
%       Intercept  - mean difference from z (fixed effect intercept)
%       pValue     - p-value for test if mean > z
%
%   Requires: Statistics and Machine Learning Toolbox (fitlme)

    % Dimensions
    [N, T] = size(X);
    
    % Ensure A is categorical
    A = categorical(A(:));
    
    if numel(A) ~= N
        error('Length of A (%d) must match number of rows in X (%d).', numel(A), N);
    end
    
    % Preallocate
    intercepts = nan(T,1);
    pvals      = nan(T,1);
    
    % Loop over timepoints
    for t = 1:T
        % Data table for timepoint t
        Xt  = X(:,t);
        tbl = table(A, Xt, 'VariableNames', {'Subject','Value'});
        
        % Center relative to z
        tbl.ValueCentered = tbl.Value - z;
        
        % Mixed-effects model with random intercept per subject
        lme = fitlme(tbl, 'ValueCentered ~ 1 + (1|Subject)');
        
        % Extract intercept and p-value
        fe = fixedEffects(lme);       % intercept estimate
        stats = lme.Coefficients;     % table of estimates, SEs, p-values
        intercepts(t) = fe;
        pvals(t)      = stats.pValue(1);
    end
    
    % Output table
    results = table((1:T)', intercepts, pvals, ...
        'VariableNames', {'Time','Intercept','pValue'});
end
