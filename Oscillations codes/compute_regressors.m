function [regressors, theta_matrix, rep_beh_regressor, partner_beh_regressor]  = compute_regressors(psth_struct, prepost_intervals,repeated_animal, onset_bool )
% COMPUTE_REGRESSORS
% Generates bout-aligned regressors (0/1 with peri-bout extension) and extracts
% corresponding theta power from psth_struct.play_bout_onset.
%
% Inputs:
%   psth_struct        - struct from GENERATE_THETA_PSTH (single channel)
%   prepost_intervals  - [pre_bout_interval, post_bout_interval] in seconds
%
% Outputs:
%   regressors   - [n_bouts x n_bins] matrix, 1 = in-bout, 0 = peri-bout, NaN = overlapping intervals
%   theta_matrix - [n_bouts x n_bins] matrix of theta power for same intervals

%% Parameters
pre_bout  = prepost_intervals(1);
post_bout = prepost_intervals(2);
hist_range = psth_struct.hist_range;      % e.g., [-20 20]
bin_size = diff(hist_range) / (size(psth_struct.play_bout_onset,2) - 1);  
time_bins = linspace(hist_range(1), hist_range(2), size(psth_struct.play_bout_onset,2));
n_bins    = numel(time_bins);
Behavior = psth_struct.Behavior;

ramp_time = 0.5; % ramp duration in seconds
ramp_bins = round(ramp_time / bin_size);
%% Safety check: max bout length + post_bout must fit in hist_range
bout_lengths = psth_struct.play_bouts_table(:,2) - psth_struct.play_bouts_table(:,1);
max_bout_len = max(bout_lengths);
if (max_bout_len + post_bout) > (hist_range(2) - hist_range(1))
    error('Bout length + post_bout exceeds hist_range. Increase hist_range or reduce intervals.');
end

%% Init outputs
n_bouts = size(psth_struct.play_bouts_table,1);
regressors   = nan(n_bouts, n_bins);
theta_matrix = nan(n_bouts, n_bins);
rep_beh_regressor = nan(n_bouts, n_bins);
partner_beh_regressor = nan(n_bouts, n_bins);

%% Iterate bouts
for i = 1:n_bouts
    bout_start = psth_struct.play_bouts_table(i,1);
    bout_end   = psth_struct.play_bouts_table(i,2);

    % Neighbor bouts
    if i > 1
        prev_end = psth_struct.play_bouts_table(i-1,2);
    else
        prev_end = -Inf;
    end
    if i < n_bouts
        next_start = psth_struct.play_bouts_table(i+1,1);
    else
        next_start = Inf;
    end

    % Absolute limits for peri-bout interval
    pre_limit  = max(bout_start + pre_bout, bout_start - 0.5*(bout_start - prev_end));
    post_limit = min(bout_end + post_bout,  bout_end + 0.5*(next_start - bout_end));

    % Convert to relative (bout onset at 0)
    rel_pre_start  = max(pre_limit - bout_start, hist_range(1));
    rel_post_end   = min(post_limit - bout_start, hist_range(2));
    rel_bout_start = 0;  
    rel_bout_end   = min(bout_end - bout_start, hist_range(2)); 

    % Indices for peri-bout window
    peri_idx = find(time_bins >= rel_pre_start & time_bins <= rel_post_end);
    bout_idx = find(time_bins >= rel_bout_start & time_bins <= rel_bout_end);

    if isempty(peri_idx)
        continue; % Skip if out of bounds
    end

    % Fill regressor: 0 in peri-bout window, 1 in bout
    reg_vec = nan(1, n_bins);
    reg_vec(peri_idx) = 0;
    reg_vec(bout_idx) = 1;

    % Assign to outputs
    regressors(i,:)   = reg_vec;
    theta_matrix(i,:) = psth_struct.play_bout_onset(i,:);

     % --- Behavior regressors ---

    % Find first behavior start per animal during bout interval
    bout_abs_start = bout_start;
    bout_abs_end = bout_end;

    % Recorded animal behaviors
    rep_beh_idx = find(strcmp(Behavior.Animal, repeated_animal) & ...
        Behavior.Start >= bout_abs_start & Behavior.Start <= bout_abs_end);
    if ~isempty(rep_beh_idx)
        first_rep_beh_start = Behavior.Start(rep_beh_idx(1));
    else
        first_rep_beh_start = NaN;
    end

    % Partner animal behaviors
    partner_idx = find(~strcmp(Behavior.Animal, repeated_animal) & ...
        Behavior.Start >= bout_abs_start & Behavior.Start <= bout_abs_end);
    if ~isempty(partner_idx)
        first_partner_beh_start = Behavior.Start(partner_idx(1));
    else
        first_partner_beh_start = NaN;
    end

    % Initialize regressors for behaviors as zeros in peri_idx range
    rep_beh_regressor(i, :) = nan(1,n_bins);
    rep_beh_regressor(i, peri_idx) = 0;
    partner_beh_regressor(i, :) = nan(1,n_bins);
    partner_beh_regressor(i, peri_idx) = 0;

    % Ramp for repeated animal
    if ~isnan(first_rep_beh_start)
        rel_beh_bin = find(time_bins >= (first_rep_beh_start - bout_start), 1, 'first');
        ramp_end_bin = min(rel_beh_bin + ramp_bins - 1, peri_idx(end));
        ramp_bins_indices = rel_beh_bin:ramp_end_bin;
        n_ramp = numel(ramp_bins_indices);
        x = linspace(0, pi, n_ramp);
        ramp_vals = (1 - cos(x)) / 2;  % cosine ramp 0->1
        rep_beh_regressor(i, ramp_bins_indices) = ramp_vals;
    end

    % Ramp for partner animal
    if ~isnan(first_partner_beh_start)
        rel_beh_bin = find(time_bins >= (first_partner_beh_start - bout_start), 1, 'first');
        ramp_end_bin = min(rel_beh_bin + ramp_bins - 1, peri_idx(end));
        ramp_bins_indices = rel_beh_bin:ramp_end_bin;
        n_ramp = numel(ramp_bins_indices);
        x = linspace(0, pi, n_ramp);
        ramp_vals = (1 - cos(x)) / 2;  % cosine ramp 0->1
        partner_beh_regressor(i, ramp_bins_indices) = ramp_vals;
    end

end

end
