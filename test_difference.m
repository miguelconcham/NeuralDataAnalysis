function [p, real_E, perm_E] = test_difference(X, Y, n_perm)
% test_difference - Tests whether two multivariate distributions differ using energy distance
%
%   [p, real_E, perm_E] = test_difference(X, Y, n_perm)
%
% Inputs:
%   X, Y    - Two matrices of size [n_samples × n_features]
%   n_perm  - (Optional) Number of permutations (default = 1000)
%
% Outputs:
%   p       - p-value of the test
%   real_E  - observed energy distance
%   perm_E  - vector of energy distances from permuted data

    if nargin < 3
        n_perm = 1000;
    end

    % Compute observed energy distance
    real_E = energy_distance(X, Y);

    % Combine data
    combined = [X; Y];
    nX = size(X, 1);
    n_total = size(combined, 1);
    perm_E = zeros(n_perm, 1);

    % Permutation test
    for i = 1:n_perm
        idx = randperm(n_total);
        X_perm = combined(idx(1:nX), :);
        Y_perm = combined(idx(nX+1:end), :);
        perm_E(i) = energy_distance(X_perm, Y_perm);
    end

    % Calculate p-value (right-tailed)
    p = mean(perm_E >= real_E);
end

function E = energy_distance(X, Y)
% Computes the energy distance between X and Y

    n = size(X, 1);
    m = size(Y, 1);

    Dxy = pdist2(X, Y);      % Cross-distance
    Dxx = pdist2(X, X);      % Intra-group X
    Dyy = pdist2(Y, Y);      % Intra-group Y

    E = 2*mean(Dxy(:)) - mean(Dxx(~eye(n))) - mean(Dyy(~eye(m)));
end