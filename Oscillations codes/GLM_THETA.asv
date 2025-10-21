saving_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\Theta psth';

load([saving_folder,'\psth_structure.mat'],'psth_structure');
load([saving_folder,'\animal_names.mat'],'animal_names');


animal_index                = [];
all_regressors              = [];
all_theta_matrix            = [];
all_rep_beh_regressor       = [];
all_partner_beh_regressor   = [];
session_info                = [];
time = psth_ranges(1):bin_size:psth_ranges(2)+bin_size;
smooth_wind = 20;
baseline_range = [-2 0];
baseline_index = time<baseline_range(2) & time>baseline_range(1);
all_play_bout_lengths = [];
extra_time = [-5 5];
for fn = 1:numel(psth_structure)
    animal_info = animal_names{fn,1};
    animal_info = strsplit(animal_info, ' ');
    animal_name = animal_info{1};
    session     =  animal_info{2};
    repeated_animal = animal_info{end};
    [~, theta_matrix_pre, ~, ~] = compute_regressors(psth_structure(fn), extra_time, repeated_animal);

     [regressors, theta_matrix, rep_beh_regressor, partner_beh_regressor]  = compute_regressors_offset(psth_structure(fn), extra_time, repeated_animal);
    all_regressors              = [all_regressors;regressors];
   


        for trial=1:size(theta_matrix,1)
            theta_matrix(trial,:) = ( theta_matrix(trial,:) - mean( theta_matrix_pre(trial,baseline_index)))/std( theta_matrix_pre(trial,baseline_index));
            theta_matrix(trial,:) = movmean(theta_matrix(trial,:), smooth_wind);
        end
      all_theta_matrix            = [all_theta_matrix;theta_matrix];

    all_rep_beh_regressor       = [all_rep_beh_regressor;rep_beh_regressor];
    all_partner_beh_regressor   = [all_partner_beh_regressor;partner_beh_regressor];
    session_info                = [session_info; repmat({animal_name,session,animal_names{fn,2}} ,size(regressors,1),1)];
    all_play_bout_lengths       = [all_play_bout_lengths;diff(psth_structure(fn).play_bouts_table')'];
end

%%
x_lim = [-10 10];
bin_size = psth_structure(1).wind_length - psth_structure(1).wind_overlap;
psth_ranges = psth_structure(1).hist_range;

regressors_norm = (all_regressors);
beh_regressor_norm = (all_rep_beh_regressor);
partner_beh_regressor_norm = (all_partner_beh_regressor);
% 
% regressors_norm = l2_normalize_regressors(all_regressors);
% beh_regressor_norm = l2_normalize_regressors(all_rep_beh_regressor);
% partner_beh_regressor_norm = l2_normalize_regressors(all_partner_beh_regressor);

[all_play_bout_lengths_sorted, order] = sort(all_play_bout_lengths);
all_theta_matrix(isnan(all_regressors)) = NaN;
 figure
 subplot(1,4,1)
 imagesc(time, 1:numel(all_play_bout_lengths), regressors_norm(order,:)+1)
 % clim([0 2])
 axis xy
 xlim(x_lim)

 subplot(1,4,2)
 imagesc(time, 1:numel(all_play_bout_lengths),beh_regressor_norm(order,:)+1)
 % clim([0 2])
 axis xy
 xlim(x_lim)
 subplot(1,4,3)
 imagesc(time, 1:numel(all_play_bout_lengths),partner_beh_regressor_norm(order,:)+1)
 % clim([0 2])
 axis xy
 xlim(x_lim)

 subplot(1,4,4)
 imagesc(time, 1:numel(all_play_bout_lengths),all_theta_matrix(order,:))
  axis xy
 xlim(x_lim)
  %% now ploting thete



length_limit =2;
resampled_all_theta_matrix = all_theta_matrix;
resampled_all_theta_matrix(repmat(time,size(all_theta_matrix,1),1)<0 & isnan(all_regressors)) = NaN;;
indextinclude = all_play_bout_lengths>=length_limit;
  figure
  array = resampled_all_theta_matrix(indextinclude,:);
  [~, ~, ci] = ttest(array);

  no_nan_ci = ~any(isnan(ci));
  fill([time(no_nan_ci) fliplr(time(no_nan_ci))], [ci(1,no_nan_ci) fliplr(ci(2,no_nan_ci))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
  hold on
  plot(time,mean(array, 'omitmissing'), 'k')


xlim([-5 extra_time(2)])



  %%


length_limit =1;

indextinclude = all_play_bout_lengths>=length_limit;
power = all_theta_matrix(indextinclude,:);

subject_idx =session_info(indextinclude,1);
time_range = [-.5 3];
limted_time = time;
valid_bins = find(limted_time >= time_range(1) & limted_time <= time_range(2));
power = power(:,valid_bins);
limted_time = limted_time(valid_bins);

[nTrials, nTime] = size(power);

% Preallocate
est  = nan(nTime,1);
se   = nan(nTime,1);
pvals = nan(nTime,1);
d    = nan(nTime,1);
ci   = nan(nTime,2);

% Reshape to long format
[trial_idx, time_idx] = ndgrid(1:nTrials,1:nTime);
tbl = table;
tbl.Power = power(:);
tbl.Subject = categorical(subject_idx(trial_idx(:)));
time_matrix = repmat(limted_time, nTrials, 1);
tbl.Time = time_matrix(:);

% Loop over bins
for i = 1:nTime
    tbl_t = tbl(tbl.Time == limted_time(i),:);
    lme = fitlme(tbl_t,'Power ~ 1 + (1|Subject)');

    est(i) = lme.Coefficients.Estimate(1);
    se(i)  = lme.Coefficients.SE(1);
    pvals(i) = lme.Coefficients.pValue(1);
    ci_t = coefCI(lme);
    ci(i,:) = ci_t(1,:);

    % Cohen's d-like standardized effect size
    sd_within = std(tbl_t.Power, 'omitmissing');
    d(i) = est(i) / sd_within;
end

% Multiple comparison correction (FDR)
pvals_fdr = mafdr(pvals,'BHFDR',true);

% Package results
results.time = limted_time(:);
results.est = est;
results.se = se;
results.ci = ci;
results.pvals = pvals;
results.pvals_fdr = pvals_fdr;
results.d = d;

%%


alpha = 0.05;


limited_time = results.time;
est = results.est;
ci = results.ci;
pvals_fdr = results.pvals_fdr;
d = results.d;

figure;
subplot(2,1,1); hold on;

% Shaded CI
fill([limited_time; flipud(limited_time)], [ci(:,1); flipud(ci(:,2))], ...
    [0.8 0.8 1], 'EdgeColor','none','FaceAlpha',0.4);
plot(limited_time, est, 'b','LineWidth',2);

% Mark significant bins
sig_idx = pvals_fdr < alpha;
plot(limited_time(sig_idx), est(sig_idx), 'r*','MarkerSize',6);

ylabel('Mean Power');
title('Mixed-Effects Power (CI + FDR-corrected sig)');
grid on;

subplot(2,1,2);
plot(limited_time, d, 'k','LineWidth',2);
ylabel('Effect size (Cohen''s d)');
xlabel('Time (s)');
grid on;


   %%
n_bins = size(regressors_norm,2);
   regressor_table = [regressors_norm(~isnan(regressors_norm)) beh_regressor_norm(~isnan(beh_regressor_norm)) partner_beh_regressor_norm(~isnan(partner_beh_regressor_norm))];

 animal_col = repmat(session_info(:,1), 1, n_bins);   % NxT
animal_col =animal_col( ~isnan(regressors_norm));
session_col = repmat(session_info(:,2), 1, n_bins);
session_col = session_col( ~isnan(regressors_norm));
electrode_col = repmat(cell2mat(session_info(:,3)), 1, n_bins);
electrode_col = electrode_col( ~isnan(regressors_norm));

tbl = table(all_theta_matrix(~isnan(all_theta_matrix)), regressors_norm(~isnan(regressors_norm)), beh_regressor_norm(~isnan(beh_regressor_norm)), partner_beh_regressor_norm(~isnan(partner_beh_regressor_norm)), ...
    categorical(animal_col), categorical(session_col), categorical(electrode_col), ...
    'VariableNames', {'theta','PlayBout','SelfPlay','OtherPlay','animal','session','electrode'});

lme_full = fitlme(tbl(:, {'theta','PlayBout','SelfPlay','OtherPlay','animal','session','electrode'}), ...
    'theta ~ PlayBout + SelfPlay + OtherPlay + (1|animal) + (1|animal:session) + (1|animal:session:electrode)');

lme_other = fitlme(tbl(:, {'theta','PlayBout','OtherPlay','animal','session','electrode'}), ...
    'theta ~ PlayBout  + OtherPlay + (1|animal) + (1|animal:session) + (1|animal:session:electrode)');

lme_self = fitlme(tbl(:, {'theta','PlayBout','SelfPlay','animal','session','electrode'}), ...
    'theta ~ PlayBout  + SelfPlay + (1|animal) + (1|animal:session) + (1|animal:session:electrode)');


lme_self_pure = fitlme(tbl(:, {'theta','SelfPlay','animal','session','electrode'}), ...
    'theta ~ SelfPlay + (1|animal) + (1|animal:session) + (1|animal:session:electrode)');

lme_PlayBout = fitlme(tbl(:, {'theta','PlayBout','animal','session','electrode'}), ...
    'theta ~ PlayBout   + (1|animal) + (1|animal:session) + (1|animal:session:electrode)');

%%
deviance_simpler = lme_PlayBout.ModelCriterion.Deviance;  % Deviance of simpler model (n params)
deviance_complex =  lme_other.ModelCriterion.Deviance;  % Deviance of complex model (m params)
n = 1;   % Number of parameters in simpler model
m = 2;   % Number of parameters in complex model

% Calculate deviance difference
D = deviance_simpler - deviance_complex;

% Degrees of freedom difference
df = m - n;

% Compute p-value from chi-square distribution
p_other = 1 - chi2cdf(D, df);


deviance_simpler = lme_other.ModelCriterion.Deviance;  % Deviance of simpler model (n params)
deviance_complex =  lme_full.ModelCriterion.Deviance;  % Deviance of complex model (m params)
n = 1;   % Number of parameters in simpler model
m = 3;   % Number of parameters in complex model

% Calculate deviance difference
D = deviance_simpler - deviance_complex;

% Degrees of freedom difference
df = m - n;

% Compute p-value from chi-square distribution
p_other_full = 1 - chi2cdf(D, df);


deviance_simpler = lme_PlayBout.ModelCriterion.Deviance;  % Deviance of simpler model (n params)
deviance_complex =  lme_self.ModelCriterion.Deviance;  % Deviance of complex model (m params)
n = 1;   % Number of parameters in simpler model
m = 2;   % Number of parameters in complex model

% Calculate deviance difference
D = deviance_simpler - deviance_complex;

% Degrees of freedom difference
df = m - n;

% Compute p-value from chi-square distribution
p_self = 1 - chi2cdf(D, df);


deviance_simpler = lme_self.ModelCriterion.Deviance;  % Deviance of simpler model (n params)
deviance_complex =  lme_full.ModelCriterion.Deviance;  % Deviance of complex model (m params)
n = 1;   % Number of parameters in simpler model
m = 3;   % Number of parameters in complex model

% Calculate deviance difference
D = deviance_simpler - deviance_complex;

% Degrees of freedom difference
df = m - n;

% Compute p-value from chi-square distribution
p_self_full = 1 - chi2cdf(D, df);





deviance_simpler = lme_self_pure.ModelCriterion.Deviance;  % Deviance of simpler model (n params)
deviance_complex =  lme_self.ModelCriterion.Deviance;  % Deviance of complex model (m params)
n = 1;   % Number of parameters in simpler model
m = 2;   % Number of parameters in complex model

% Calculate deviance difference
D = deviance_simpler - deviance_complex;

% Degrees of freedom difference
df = m - n;

% Compute p-value from chi-square distribution
p_pure_self = 1 - chi2cdf(D, df);

%%


figure

semilogy([1 2 3 4 5],[p_other p_other_full  p_pure_self p_self p_self_full] )
hold on
semilogy([1 2 3 4 5],ones(1,5) )
