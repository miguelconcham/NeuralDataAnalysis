

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

psth_structure = [];
wind_length     = 1;
wind_overlap    = .990;
min_separation = .200;
f               = .1:.1:6;
freq_pow_range  = [.5 5];

%%
for fn = 1:numel(animal_list)

    if fn==1
        psth_structure = GENERATE_THETA_PSTH([npx_Raw_Data, '\', animal_list(fn).name],wind_length,wind_overlap,min_separation,f,freq_pow_range )
        n_strctut = n_strctut+numel(psth_structure);
        animal_names = [animal_names;[repmat(animal_list(fn).name,numel(psth_structure),1) num2cell(1:numel(psth_structure))']]
    else
        transt_psth = GENERATE_THETA_PSTH([npx_Raw_Data, '\', animal_list(fn).name],wind_length,wind_overlap,min_separation,f,freq_pow_range )
      
        for sub_j=1:numel(transt_psth)
    
            psth_structure(n_strctut) = transt_psth(sub_j);
            n_strctut = n_strctut+1;
        end
        animal_names = [animal_names;[repmat({animal_list(fn).name},numel(transt_psth),1) num2cell(1:numel(transt_psth))' ]]

    end


end
%% save if needed
disp('saving')
% save([saving_folder,'\psth_structure_delta.mat'],'psth_structure', '-v7.3');
% save([saving_folder,'\animal_names_delta.mat'],'animal_names');

%% load if needed
disp('loading')
load([saving_folder,'\psth_structure_delta.mat'],'psth_structure');
load([saving_folder,'\animal_names_delta.mat'],'animal_names');
disp('ready')
%% merging_psth
smooth_wind = 20;
baseline_range = [-2 0]
animal_label = {'B1D1','B1S3','B2S2','B3D2', 'B4S2', 'B4D4'};
electorde_numner = [1 2];
bin_size = psth_structure(1).wind_length - psth_structure(1).wind_overlap;
psth_ranges = psth_structure(1).hist_range;
wrap_range = psth_structure(1).range_time_wrap;
time = psth_ranges(1):bin_size:psth_ranges(2)+bin_size;
baseline_index = time<baseline_range(2) & time>baseline_range(1);
baseline_index_time_wrap = 1:round((abs(wrap_range(1))/bin_size));
all_psth_onset                  = [];
all_psth_onset_behavior         = [];
all_psth_onset_only_playobut    = [];
all_psth_offset         = [];
all_psth_tw             = [];
all_psth_tw_3points     = [];
all_play_bouts          = [];
time_wrap_time          = [(baseline_index_time_wrap*bin_size) + wrap_range(1),linspace(0,1,psth_structure(1).n_bins_time_wrap),1 + (1:round((abs(wrap_range(2))/bin_size)))*bin_size];
time_wrap_3_points      = [(baseline_index_time_wrap*bin_size) + wrap_range(1),linspace(0,1-1/psth_structure(1).n_bins_time_wrap,psth_structure(1).n_bins_time_wrap), ...
   linspace(1,2-1/psth_structure(1).n_bins_time_wrap,psth_structure(1).n_bins_time_wrap),2 + (1:round((abs(wrap_range(2))/bin_size)))*bin_size];

session_index = [];
animal_index = [];
for j=1:numel(psth_structure)

    if contains(animal_names{j},animal_label)
            
        animal_num      = find(cell2mat(cellfun(@(x) contains(animal_names{j},x), animal_label, 'UniformOutput',false)));  
        electrode_num   = find(cell2mat(cellfun(@(x) contains(animal_names{j},x), animal_label, 'UniformOutput',false))); 
        this_animal_playbouts = psth_structure(j).play_bouts_table;
        this_animal_lengths = diff(this_animal_playbouts');
        all_play_bouts = [all_play_bouts;this_animal_playbouts];

        this_psth_onset         = psth_structure(j).play_bout_onset;
        this_psth_onset_onlypb  = this_psth_onset;
        animal_index = [animal_index;repmat(animal_num,size(this_psth_onset,1),1)];
        session_index = [session_index;repmat(j,size(this_psth_onset,1),1)];
        for trial=1:size(this_psth_onset,1)
            this_psth_onset(trial,:) = ( this_psth_onset(trial,:) - mean( this_psth_onset(trial,baseline_index)))/std( this_psth_onset(trial,baseline_index));
            this_psth_onset(trial,:) = movmean(this_psth_onset(trial,:), smooth_wind);
            % this_psth_onset_onlypb(trial,:) = this_psth_onset(trial,:);
            this_psth_onset(trial,time> this_animal_lengths(trial)) = NaN;
        end
        all_psth_onset      = [all_psth_onset; this_psth_onset];
        all_psth_onset_only_playobut = [all_psth_onset_only_playobut; this_psth_onset_onlypb];

        this_psth_onset     = psth_structure(j).play_bout_onset;
        this_psth_offset    = psth_structure(j).play_bout_offset;
         for trial=1:size(this_psth_offset,1)
            this_psth_offset(trial,:) = ( this_psth_offset(trial,:) - mean( this_psth_onset(trial,baseline_index)))/std( this_psth_onset(trial,baseline_index));
            this_psth_offset(trial,:) = movmean(this_psth_offset(trial,:), smooth_wind);
         end         
        all_psth_offset = [all_psth_offset; this_psth_offset];

        this_psth_tw = psth_structure(j).play_bout_tw_this;
        for trial=1:size(this_psth_tw,1)
            this_psth_tw(trial,:) = ( this_psth_tw(trial,:) - mean( this_psth_tw(trial,baseline_index_time_wrap)))/std( this_psth_tw(trial,baseline_index_time_wrap));
        end
        all_psth_tw = [all_psth_tw; this_psth_tw];


         this_psth_tw = psth_structure(j).three_point_tw;
        for trial=1:size(this_psth_tw,1)
            this_psth_tw(trial,:) = ( this_psth_tw(trial,:) - mean( this_psth_tw(trial,baseline_index_time_wrap)))/std( this_psth_tw(trial,baseline_index_time_wrap));
        end
        all_psth_tw_3points = [all_psth_tw_3points; this_psth_tw];



        this_psth_ab = psth_structure(j).animal_behavior_onset;
        for trial=1:size(this_psth_ab,1)
            this_psth_ab(trial,:) = ( this_psth_ab(trial,:) - mean( this_psth_ab(trial,baseline_index_time_wrap)))/std( this_psth_ab(trial,baseline_index_time_wrap));
            this_psth_ab(trial,:) = movmean(this_psth_ab(trial,:), smooth_wind);
             this_psth_ab(trial,time> this_animal_lengths(trial)) = NaN;
        end
        all_psth_onset_behavior = [all_psth_onset_behavior; this_psth_ab];

    end
end

play_bout_length = diff(all_play_bouts')';


[sorted_play_bout_length, order] = sort(play_bout_length);

%%

X_lim = [baseline_range(1) 3];

figure

time2use = time;
for an= 1:numel(animal_label)

    [sorted_play_bout_length, order] = sort(play_bout_length(animal_index==an,:));
    subplot(5,numel(animal_label),(1:numel(animal_label):2*numel(animal_label)) + an-1)
    array = all_psth_onset_behavior(animal_index==an,:);
    imagesc(time2use,1:numel(sorted_play_bout_length),array(order,:) )
    xlim(X_lim)
    clim([-2 2])
    axis xy

    subplot(5,numel(animal_label),((2*numel(animal_label) + 1):numel(animal_label):5*numel(animal_label)) + an-1)

    [~, ~, ci]  = ttest(array);
    fill([time2use fliplr(time2use)], [ci(1,:) fliplr(ci(2,:))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time2use,mean(array, 'omitmissing'), 'k')



    xlim(X_lim)
end


%%
figure
min_length = .0;
for an= 1:numel(animal_label)
    animal_bool = animal_index==an;
    length_bool = play_bout_length>min_length;
    [sorted_play_bout_length, order] = sort(play_bout_length(animal_bool & length_bool,:));
    subplot(5,numel(animal_label),(1:numel(animal_label):2*numel(animal_label)) + an-1)

    array = all_psth_onset_only_playobut(animal_bool & length_bool,:);
    imagesc(time,1:numel(sorted_play_bout_length),array(order,:) )
    xlim(X_lim)
    clim([-2 2])
    axis xy
    hold on
    plot([0 0],[1 numel(sorted_play_bout_length)], 'w')
    plot(sorted_play_bout_length,1:numel(sorted_play_bout_length), 'w')
    title(animal_label{an})

    subplot(5,numel(animal_label),((2*numel(animal_label) + 1):numel(animal_label):5*numel(animal_label)) + an-1)

    [~, ~, ci]  = ttest(array);
    no_nan = ~any(isnan(ci));
    fill([time(no_nan) fliplr(time(no_nan))], [ci(1,no_nan) fliplr(ci(2,no_nan))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time,mean(array, 'omitmissing'), 'k')
    xlim(X_lim)
end

%% onst to behavior

figure
min_length = .0;

for an= 1:numel(animal_label)
    animal_bool = animal_index==an;
    length_bool = play_bout_length>min_length;
    [sorted_play_bout_length, order] = sort(play_bout_length(animal_bool & length_bool,:));
    subplot(5,numel(animal_label),(1:numel(animal_label):2*numel(animal_label)) + an-1)

    array = all_psth_onset_behavior(animal_bool & length_bool,:);
    imagesc(time,1:numel(sorted_play_bout_length),array(order,:) )
    xlim(X_lim)
    clim([-2 2])
    axis xy
    hold on
    plot([0 0],[1 numel(sorted_play_bout_length)], 'w')
    plot(sorted_play_bout_length,1:numel(sorted_play_bout_length), 'w')
    title(animal_label{an})

    subplot(5,numel(animal_label),((2*numel(animal_label) + 1):numel(animal_label):5*numel(animal_label)) + an-1)

    [~, ~, ci]  = ttest(array);
    fill([time fliplr(time)], [ci(1,:) fliplr(ci(2,:))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time,mean(array, 'omitmissing'), 'k')
    xlim(X_lim)
end

%% timewrpa playbout and play behavior

         
X_lim = [-2 4]

figure
min_length = .0;

for an= 1:numel(animal_label)
    animal_bool = animal_index==an;
    length_bool = play_bout_length>min_length;
    [sorted_play_bout_length, order] = sort(play_bout_length(animal_bool & length_bool,:));
    subplot(5,numel(animal_label),(1:numel(animal_label):2*numel(animal_label)) + an-1)

    array = all_psth_tw_3points(animal_bool & length_bool,:);
    imagesc(time_wrap_3_points,1:numel(sorted_play_bout_length),array(order,:) )
    xlim(X_lim)
    clim([-2 2])
    axis xy
    hold on
    plot([0 0],[1 numel(sorted_play_bout_length)], 'w')
    plot(sorted_play_bout_length,1:numel(sorted_play_bout_length), 'w')
    title(animal_label{an})

    subplot(5,numel(animal_label),((2*numel(animal_label) + 1):numel(animal_label):5*numel(animal_label)) + an-1)

    [~, ~, ci]  = ttest(array);
    fill([time_wrap_3_points fliplr(time_wrap_3_points)], [ci(1,:) fliplr(ci(2,:))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time_wrap_3_points,mean(array, 'omitmissing'), 'k')
    xlim(X_lim)
end


%%
min_length = 0;
length_bool = play_bout_length>min_length;
figure
    array = all_psth_tw_3points( length_bool,:);

    [~, ~, ci]  = ttest(array);
    fill([time_wrap_3_points fliplr(time_wrap_3_points)], [ci(1,:) fliplr(ci(2,:))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time_wrap_3_points,mean(array, 'omitmissing'), 'k')
    xlim(X_lim)



%%

figure
hold on
array = all_psth_onset_behavior;

 [~, ~, ci]  = ttest(array);
 fill([time fliplr(time)], [ci(1,:) fliplr(ci(2,:))], 'r', 'FaceAlpha',.25, 'EdgeColor','none')
plot(time,mean(array, 'omitmissing'), 'r')
hold on

array = all_psth_onset;
 [~, ~, ci]  = ttest(array);
 fill([time fliplr(time)], [ci(1,:) fliplr(ci(2,:))], 'b', 'FaceAlpha',.25, 'EdgeColor','none')
plot(time,mean(array, 'omitmissing'), 'b')
plot(time, mean(all_psth_onset), 'b')
 xlim(X_lim)
%% estimating mixed model per time

length_limit =0;
array2use = all_psth_tw_3points;
indextinclude = play_bout_length>=length_limit & ~any(isnan(all_psth_onset_behavior),2);
% power = all_psth_onset_behavior(indextinclude,:);
power = array2use(indextinclude,:);
zscore_limit = 4;
subject_idx =animal_index(indextinclude);
time_range = [-2 4];
% limted_time = time;
limted_time =time_wrap_3_points;
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
length_matrix = repmat(play_bout_length(indextinclude),1,numel(limted_time));
tbl.length = length_matrix(:);

tbl(abs(tbl.Power)>zscore_limit,:) = []; %remove outliers if needed
tbl(isnan(tbl.Power),:) = [];
subject_list = unique(tbl.Subject);
animal_counts_per_bin = nan(nTime,numel(subject_list));
re = nan(nTime,numel(subject_list));

% Loop over bins
for i = 1:nTime
    tbl_t = tbl(tbl.Time == limted_time(i) & tbl.Time<tbl.length,:);
   [animal_this_bin, id] = groupcounts(tbl_t.Subject);
   animal_counts_per_bin(i,ismember(subject_list,id )) = animal_this_bin;
     if min(animal_this_bin)>10
    lme = fitlme(tbl_t,'Power ~ 1 + (1|Subject)');
    re(i,:) = randomEffects(lme);

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
results.zscore_limit = zscore_limit;
results.time_range = time_range;
play_song([],[],[])
%% save if needed

save([saving_folder,'\results_play_bout_tw3points_zscore4.mat'],'results');

%% load if needed
% load([saving_folder,'\results_play_bout.mat'],'results');
load([saving_folder,'\results_play_bout_delta_all_onlybout.mat'],'results');

%%


alpha = 0.05;


limited_time = results.time;
est = results.est;
ci = results.ci;
% pvals_fdr = results.pvals_fdr;
pvals_fdr = results.pvals;
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

alpha = 0.05;
[cluster_pvals, clusters, cluster_stats, perm_max_stats] = cluster_perm_test(array(hmm_type==1,:), array(hmm_type==0,:), alpha, 500);

save([saving_folder,'\cluster_pvals.mat'],'cluster_pvals','clusters','cluster_stats','perm_max_stats', 'array', 'hmm_type','alpha', 'time')

%% delta onset per session

all_animals_delta_onset = [];
for j=1:numel(psth_structure)

    this_session = session_index==j;

all_animals_delta_onset = [all_animals_delta_onset;mean(all_psth_onset(this_session,:), 'omitmissing')];
end



%%

figure
plot(time,all_animals_delta_onset, 'k:')
hold on
[~, ~, ci] = ttest(all_animals_delta_onset);

plot(time, mean(all_animals_delta_onset), 'k')
xlim([-3 4])