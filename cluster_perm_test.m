function [cluster_pvals, clusters, cluster_stats, perm_max_stats] = cluster_perm_test(data1, data2, alpha, nPerm)
% Performs cluster-based permutation test between two groups of time series data
% Inputs:
%   data1, data2: matrices [nObservations x nTimePoints]
%   alpha: cluster-forming threshold on p-values (e.g., 0.05)
%   nPerm: number of permutations (e.g., 1000)
%
% Outputs:
%   cluster_pvals: p-values for each observed cluster
%   clusters: cell array of indices for each cluster
%   cluster_stats: cluster statistics (sum of t-values) for each cluster
%   perm_max_stats: max cluster stats from each permutation (for plotting / diagnostics)

[n1, nTime] = size(data1);
[n2, ~] = size(data2);
allData = [data1; data2];
labels = [ones(n1,1); 2*ones(n2,1)];
  [~, p, ~, stats] = ttest2(data1, data2, 'Vartype','unequal');


% Step 1: Compute observed t- and p-values at each time point
tvals = stats.tstat;
pvals = p;


% Step 2: Define clusters based on p-values < alpha
cluster_mask = pvals < alpha;
d = diff([0 cluster_mask 0]);
cluster_starts = find(d == 1);
cluster_ends = find(d == -1) - 1;

clusters = cell(1, numel(cluster_starts));
cluster_stats = zeros(1, numel(cluster_starts));
for c = 1:numel(cluster_starts)
    idx = cluster_starts(c):cluster_ends(c);
    clusters{c} = idx;
    cluster_stats(c) = sum(tvals(idx));
end

% Step 3: Permutation testing
perm_max_stats = zeros(1, nPerm);
for perm = 1:nPerm
    permLabels = labels(randperm(length(labels)));
   group1 = allData(permLabels==1, :);
   group2 = allData(permLabels==2, :);
     [~, p, ~, stats] = ttest2(group1, group2, 'Vartype','unequal');
    
    perm_tvals = stats.tstat;
    perm_pvals = p;
    

    % Cluster mask using same alpha threshold on permuted p-values
    perm_mask = perm_pvals < alpha;
    d_perm = diff([0 perm_mask 0]);
    starts_perm = find(d_perm == 1);
    ends_perm = find(d_perm == -1) - 1;

    % Compute max cluster stat for this permutation
    max_stat = 0;
    for c = 1:numel(starts_perm)
        idx = starts_perm(c):ends_perm(c);
        c_stat = sum(perm_tvals(idx));
        if abs(c_stat) > abs(max_stat)
            max_stat = c_stat;
        end
    end
    perm_max_stats(perm) = max_stat;
end

% Step 4: Compute cluster p-values by comparing observed cluster stats to perm max stats
cluster_pvals = zeros(1, numel(cluster_stats));
for c = 1:numel(cluster_stats)
    cluster_pvals(c) = mean(abs(perm_max_stats) >= abs(cluster_stats(c)));
end

end
