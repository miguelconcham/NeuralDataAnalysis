

npx_Raw_Data = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\NPX data\NPX raw data';
saving_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\Theta psth';
animal_list = dir(npx_Raw_Data);
animal_list(1:2) = [];


animal_file_names =  cellfun(@(x) ['B', x],strsplit([animal_list.name], 'B'), 'UniformOutput',false)';
animal_file_names(1) = [];
% animal2exclude = {'B4D4 0826 Dual'};
animal2exclude = {''};
animal_list(ismember(animal_file_names,animal2exclude)) = [];
animal_names ={};
n_strctut = 1;
fs      = 2500;
order   = 500;
band =  [.5 5];
Hd = designfilt('bandpassfir', ...
                    'FilterOrder', order, ...
                    'CutoffFrequency1', band(1), ...
                    'CutoffFrequency2', band(2), ...
                    'SampleRate', fs, ...
                    'DesignMethod', 'window', ...
                    'Window', 'hamming');
    
psth_ranges = [-20 20];


    %%
psth_structure = [];


for fn = 1:numel(animal_list)

    if fn==1
        psth_structure = GENERATE_LOCKED_PSTH([npx_Raw_Data, '\', animal_list(fn).name],Hd,psth_ranges);
        n_strctut = n_strctut+numel(psth_structure);
        animal_names = [animal_names;[repmat(animal_list(fn).name,numel(psth_structure),1) num2cell(1:numel(psth_structure))']]
    else

        transt_psth = GENERATE_LOCKED_PSTH([npx_Raw_Data, '\', animal_list(fn).name],Hd,psth_ranges);

        for sub_j=1:numel(transt_psth)
    
            psth_structure(n_strctut) = transt_psth(sub_j);
            n_strctut = n_strctut+1;
        end
        animal_names = [animal_names;[repmat({animal_list(fn).name},numel(transt_psth),1) num2cell(1:numel(transt_psth))' ]]

    end


end
%%
disp('saving')
save([saving_folder,'\psth_structure_delta_lfp_phase.mat'],'psth_structure');
save([saving_folder,'\animal_names_delta_lfp_phase.mat'],'animal_names');


%%


load([saving_folder,'\psth_structure_delta_lfp.mat'],'psth_structure');
load([saving_folder,'\animal_names_delta_lfp.mat'],'animal_names');
%% merging_psth
smooth_wind = 20;
baseline_range = [-5 0]
animal_label = {'B1D1','B1S3','B2S2','B3D2', 'B4S2', 'B4D4'};
electorde_numner = [1 2];
bin_size = 1/fs;
psth_ranges = psth_structure(1).hist_range;
psth_ranges =psth_ranges;
time = psth_ranges(1)+bin_size:bin_size:psth_ranges(2);
baseline_index = time<baseline_range(2) & time>baseline_range(1);
all_psth_onset          = [];
animal_index            = [];
electrode_index         = [];
all_play_bouts          = [];
session_n               = [];
for j=1:numel(psth_structure)

    if contains(animal_names{j},animal_label)
        
        animal_num      = find(cell2mat(cellfun(@(x) contains(animal_names{j},x), animal_label, 'UniformOutput',false)));  
        electrode_num   = animal_names{j,2};

        this_psth_onset         = psth_structure(j).play_bout_onset;
          animal_index = [animal_index;repmat(animal_num,size(this_psth_onset,1),1)];
          electrode_index = [electrode_index;repmat(electrode_num,size(this_psth_onset,1),1)];
          session_n =   [session_n;repmat(j,size(this_psth_onset,1),1)];
          this_play_bouts = psth_structure(j).play_bouts_table;
          current_lengths =diff(this_play_bouts')';
          % for trial=1:size(this_psth_onset,1)
          %   % this_psth_onset(trial,:) = ( this_psth_onset(trial,:) - mean( this_psth_onset(trial,baseline_index), 'omitmissing'))/std( this_psth_onset(trial,baseline_index), 'omitmissing');
          %   this_psth_onset(trial,time>=current_lengths(trial)) = NaN;
          % end
        all_psth_onset      = [all_psth_onset; this_psth_onset];

        
        all_play_bouts = [all_play_bouts;psth_structure(j).play_bouts_table];


    end
end

play_bout_length = diff(all_play_bouts')';




max_z_score             =max(abs(all_psth_onset),[],2);

%%
figure
min_length = .0;
X_lim = [-1 2];
baseline_range = time<-1;
n_rand = 1000;
stacked_percentiles = [];
real_estimate       = [];

stacked_percentiles_bc = [];
real_estimate_bc       = [];
for an= 1:numel(animal_label)
    animal_bool = animal_index==an;
    length_bool = play_bout_length>min_length;
    [sorted_play_bout_length, order] = sort(play_bout_length(animal_bool & length_bool));
    subplot(5,numel(animal_label),(1:numel(animal_label):2*numel(animal_label)) + an-1)

    array = angle(all_psth_onset(animal_bool & length_bool,:));
    imagesc(time,1:numel(sorted_play_bout_length),array(order,:) )
    xlim(X_lim)
    clim([-2 2])
    axis xy
    hold on
    plot([0 0],[1 numel(sorted_play_bout_length)], 'w')
    plot(sorted_play_bout_length,1:numel(sorted_play_bout_length), 'w')
    title(animal_label{an})

    subplot(5,numel(animal_label),((2*numel(animal_label) + 1):numel(animal_label):5*numel(animal_label)) + an-1)
    array = all_psth_onset(animal_bool & length_bool,:);
    array = array./abs(array);
    surrogate_mean = nan(n_rand,size(array,2));
    surrogate_mean_bc = nan(n_rand,size(array,2));
    for nr=1:n_rand
        rand_shifts             = randsample(size(array,2),size(array,1));
        % shuffled_array          = shift_rows_conditional(array, rand_shifts, time, play_bout_length(animal_bool & length_bool));
        shuffled_array          =  circshift(array,rand_shifts);
        surrogate_mean(nr,:)    =  abs(mean(shuffled_array, 'omitmissing'));
        surrogate_mean_bc(nr,:) = (surrogate_mean(nr,:)-mean(surrogate_mean(nr,baseline_range)))/std(surrogate_mean(nr,baseline_range));
    end
        
    real_mean = abs(mean(array, 'omitmissing'));
    real_mean_bc = (real_mean-mean(real_mean(baseline_range)))/std(real_mean(baseline_range));
    T= numel(real_mean);
    percentiles = zeros(1,T);
     percentiles_bc = zeros(1,T);
     for t = 1:T
         percentiles(t) = sum(surrogate_mean(:,t) <= real_mean(t)) / n_rand * 100;
           percentiles_bc(t) = sum(surrogate_mean_bc(:,t) <= real_mean_bc(t)) / n_rand * 100;
     end
    plot(time,percentiles_bc, 'k')
    stacked_percentiles = [stacked_percentiles;percentiles];
    real_estimate       =  [real_estimate;real_mean];
    stacked_percentiles_bc = [stacked_percentiles_bc;percentiles_bc];
    real_estimate_bc       =  [real_estimate_bc;real_mean_bc];
    xlim(X_lim)
    pause(.1)
end

%%

figure
plot(time,real_estimate_bc, 'k:')
hold on
plot(time,mean(real_estimate_bc), 'k')
    xlim([-5 5])


%% now session by session
figure
stacked_percentiles = [];
real_estimate       = [];
baseline_range = time<-1;
stacked_percentiles_bc = [];
real_estimate_bc       = [];
for an= 1:numel(psth_structure)
    animal_bool = session_n==an;
    length_bool = play_bout_length>min_length;
    [sorted_play_bout_length, order] = sort(play_bout_length(animal_bool & length_bool));
    subplot(5,numel(psth_structure),(1:numel(psth_structure):2*numel(psth_structure)) + an-1)

    array = angle(all_psth_onset(animal_bool & length_bool,:));
    imagesc(time,1:numel(sorted_play_bout_length),array(order,:) )
    xlim(X_lim)
    clim([-2 2])
    axis xy
    hold on
    plot([0 0],[1 numel(sorted_play_bout_length)], 'w')
    plot(sorted_play_bout_length,1:numel(sorted_play_bout_length), 'w')
    title(num2str(an))

    subplot(5,numel(psth_structure),((2*numel(psth_structure) + 1):numel(psth_structure):5*numel(psth_structure)) + an-1)
    array = all_psth_onset(animal_bool & length_bool,:);
    array = array./abs(array);
    surrogate_mean = nan(n_rand,size(array,2));
    surrogate_mean_bc = nan(n_rand,size(array,2));
    for nr=1:n_rand
        rand_shifts             = randsample(size(array,2),size(array,1));
        % shuffled_array          = shift_rows_conditional(array, rand_shifts, time, play_bout_length(animal_bool & length_bool));
        shuffled_array          =  circshift(array,rand_shifts);
        surrogate_mean(nr,:)    =  abs(mean(shuffled_array, 'omitmissing'));
        surrogate_mean_bc(nr,:) = (surrogate_mean(nr,:)-mean(surrogate_mean(nr,baseline_range)))/std(surrogate_mean(nr,baseline_range));
    end
        
    real_mean = abs(mean(array, 'omitmissing'));
    real_mean_bc = (real_mean-mean(real_mean(baseline_range)))/std(real_mean(baseline_range));
    T= numel(real_mean);
    percentiles = zeros(1,T);
     percentiles_bc = zeros(1,T);
     for t = 1:T
         percentiles(t) = sum(surrogate_mean(:,t) <= real_mean(t)) / n_rand * 100;
           percentiles_bc(t) = sum(surrogate_mean_bc(:,t) <= real_mean_bc(t)) / n_rand * 100;
     end
    plot(time,percentiles_bc, 'k')
    stacked_percentiles = [stacked_percentiles;percentiles];
    real_estimate       =  [real_estimate;real_mean];
    stacked_percentiles_bc = [stacked_percentiles_bc;percentiles_bc];
    real_estimate_bc       =  [real_estimate_bc;real_mean_bc];
    xlim(X_lim)
    pause(.1)
end

%%





    A = real_estimate_bc(:, time<-2 & time>-4);
    B  = real_estimate_bc(:, time>0 & time<2);
% A and B: [n x T] arrays (subjects × timepoints)
[n, T] = size(A);

% Build wide-format table: columns A1...AT, B1...BT
data = [A, B];  
varNames = [ strcat("A", string(1:T)), strcat("B", string(1:T)) ];
tbl = array2table(data, 'VariableNames', varNames);
tbl.Subject = (1:n)';   % subject ID

% Within-subject design table (Condition × Time)
Condition = [repmat("A", T, 1); repmat("B", T, 1)];
Time = [ (1:T)'; (1:T)' ];
WithinDesign = table(Condition, Time);

% Fit repeated-measures model
rm = fitrm(tbl, sprintf('%s-%s ~ 1', varNames(1), varNames(end)), ...
           'WithinDesign', WithinDesign);

% Run repeated-measures ANOVA
ranovatbl = ranova(rm, 'WithinModel', 'Condition*Time');
disp(ranovatbl);


%%

figure
plot([1,2],[mean(A,2), mean(B,2)])
%%


figure
X_lim = [-2 5];
subplot(5,1,1:3)
    length_bool = play_bout_length>min_length;
    z_score_bool = max_z_score<8;
[sorted_play_bout_length, order] = sort(play_bout_length( length_bool & z_score_bool,:));

    array = all_psth_onset( length_bool & z_score_bool,:);
 imagesc(time,1:numel(sorted_play_bout_length),array(order,:) )

 xlim(X_lim)
    clim([-2 2])
    axis xy
    hold on
    plot([0 0],[1 numel(sorted_play_bout_length)], 'w')
    plot(sorted_play_bout_length,1:numel(sorted_play_bout_length), 'w')

subplot(5,1,4:5)
plot(time,mean(all_psth_onset, 'omitmissing'), 'r')


hold on

plot(time, mean(all_psth_onset), 'b')
 xlim(X_lim)
%% estimating mixed model per time

length_limit = 1;

indextinclude = play_bout_length>=length_limit & max_z_score<8;
power = all_psth_onset(indextinclude,:);
% power = all_psth_tw_3points(indextinclude,:);

subject_idx =animal_index(indextinclude);
time_range = [-2 3];
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
for i = 7776:nTime
    if mod(i,100)==0
        disp(i)
    end
    tbl_t = tbl(tbl.Time == limted_time(i),:);
     if sum(~isnan(tbl_t.Power))>10
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
play_song([],[],[])
%%

save([saving_folder,'\results_play_bout_delta_osc.mat'],'results');
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
