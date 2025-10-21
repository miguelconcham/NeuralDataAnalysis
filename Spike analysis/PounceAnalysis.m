np_data_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\NPX data\NPX raw data';
saving_directory = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\NPX data\Analysis results\Pounce analysis';
npx_folders = dir(np_data_folder);
npx_folders(1:2)    = [];
area2analyze        = 'PAG'; % either PAG, mPFC or other
behaviors2analyse   = {'Pounce_A','Pounce_Ai','Pounce_B','Pounce_Bi'};
% behaviors2analyse = {'CC','CB','CD','Pounce_A','Pounce_B','Evasion', 'Escape', 'Pin', 'Box'};
% current_dir = cd;
range_borders       = [-2 2];
bin_size            = 0.01;
pre_rate            = [-.5 0];
min_length          = 0;
ploting_range   = [-1 2];

ploting_bool = false;
all_stacked_correlations = [];
all_lengths = [];
for fn = 1:numel(npx_folders)
    disp(npx_folders(fn).name)

current_dir = [np_data_folder, '\', npx_folders(fn).name];

animal_code         = strsplit(current_dir, '\');
animal_code         = animal_code{end};
animal_code_params  = strsplit(animal_code, ' ');
animal_batch        = animal_code_params{1};
date                = animal_code_params{2};
repeated_animal     = animal_code_params{3};

[psth_tensor_onset, psth_tensor_offset,psth_edges_onset,psth_edges_offset, mean_rate_during_behavior, behavior_indexes, ~, beh_length, Behavior,ego_alter_dummy, self_animal,cluster_info] =get_psth(current_dir,behaviors2analyse,area2analyze,range_borders,bin_size);
[psth_tensor_onset_call, psth_tensor_offset_call,psth_edges_onset_call,psth_edges_offset_call, mean_rate_during_call, all_call_index, cluster_area, call_length, CallStats, call_type,calls_within,psth_calls_onset,behavior_type] = get_psth_calls(current_dir,behaviors2analyse,area2analyze,range_borders,bin_size);
cluster_index = 1:numel(cluster_area);
onset_time = .5*(psth_edges_onset(1:end-1)+psth_edges_onset(2:end));
Behavior_onset_offset = [Behavior.Start(behavior_indexes) Behavior.End(behavior_indexes)];



pre_rate_index = onset_time>=pre_rate(1) & onset_time<=pre_rate(2);
all_lengths = [all_lengths;[beh_length beh_length*0+fn]];
[sorted_length, behavior_order] = sort(beh_length);
min_length = 0;
self_index_ordered = ego_alter_dummy(behavior_order);
this_behaviort_aligned_calls = nan(size(Behavior_onset_offset,1),numel(psth_edges_onset)-1);
if exist('CallStats', 'var')    
    for bn = 1:size(Behavior_onset_offset,1)
        this_behavior_onset  = Behavior_onset_offset(bn,1);
        this_behavior_offset = Behavior_onset_offset(bn,2);

        this_behaviort_aligned_calls(bn,:) = any((psth_edges_onset(1:end-1)+this_behavior_onset>=CallStats.BeginTimes & psth_edges_onset(1:end-1)+this_behavior_onset<=CallStats.EndTimes) | ...
            (psth_edges_onset(2:end)+this_behavior_onset>=CallStats.BeginTimes & psth_edges_onset(2:end)+this_behavior_onset<=CallStats.EndTimes),1);
        this_behaviort_aligned_calls(bn,psth_edges_onset(1:end-1)>range_borders(2)+this_behavior_offset-this_behavior_onset)  = 0;
    end
end

stacked_correlations =cell(numel(cluster_index),10);
this_behaviort_aligned_calls_sorted = this_behaviort_aligned_calls(behavior_order,:);
% for cn = 1:numel(cluster_index)
for cn = 1:numel(cluster_index)
    if ploting_bool
    figure('units','normalized','outerposition',[0 0 .75 1]);
    end
    this_neuron = cluster_index(cn);
    this_neuron_area = cluster_area{this_neuron};

    

    % colormap(1-gray)
    matrix2plot = squeeze(psth_tensor_onset(this_neuron,behavior_order,:));
    matrix2plot = matrix2plot(self_index_ordered & sorted_length>min_length,:);
    this_lengths = sorted_length(self_index_ordered & sorted_length>min_length);
    matrix2images = matrix2plot;
    matrix2images(isnan(matrix2images))=0;   

    if ploting_bool
        subplot(2,3,1)
        imagesc(onset_time,1:numel(this_lengths),1-repmat(matrix2images,1,1,3));
        axis xy
        hold on
        plot((1:numel(this_lengths))*0, 1:numel(this_lengths), 'r')
        plot(this_lengths, 1:numel(this_lengths), 'r')
        plot((1:numel(this_lengths))*0 +pre_rate(1), 1:numel(this_lengths), ':r')

        xlim(ploting_range)
        yticks([])
        xticklabels([])
        ylabel('Self')
        title(['Behavior ', this_neuron_area, ' ID' num2str(this_neuron)])
    end
         
    rate_beofre_this_neuron = mean(matrix2plot(:,pre_rate_index ),2);
    call_before = mean(this_behaviort_aligned_calls_sorted(self_index_ordered & sorted_length>min_length,pre_rate_index),2);  
   
    mdl1 = fitglm(call_before, this_lengths, 'Distribution', 'gamma', 'Link', 'log');
    rate_model = fitglm(rate_beofre_this_neuron, this_lengths, 'Distribution', 'gamma', 'Link', 'log');    
    glm_pvalue_1 = rate_model.coefTest;

    mdl2 = fitglm([call_before rate_beofre_this_neuron], this_lengths, 'Distribution', 'gamma', 'Link', 'log');
    D1 = mdl1.Deviance;  % simpler model
    D2 = mdl2.Deviance;  % full model

    % Get degrees of freedom difference
    df = mdl2.NumEstimatedCoefficients - mdl1.NumEstimatedCoefficients;

    % Compute test statistic
    chi2stat = D2 - D1;

    % Compute p-value using chi-squared distribution
    p_deviance1 = 1 - chi2cdf(chi2stat, df);

    % Estiamte correaltion beteen rate and length
    [c1,p1] = corr(rate_beofre_this_neuron,this_lengths, 'Type','Spearman');
     [c1_call,p1_call] = corr(call_before,this_lengths, 'Type','Spearman');
    if ploting_bool
        subplot(2,3,2)
        swarmchart(call_before,this_lengths,'k.')
        title([c1 ,p1])

        subplot(2,3,3)

        swarmchart(call_before,this_lengths,'k.')

       
        title([c1_call ,p1_call])
    end


    % colormap(1-gray)
    matrix2plot = squeeze(psth_tensor_onset(this_neuron,behavior_order,:));
    matrix2plot = matrix2plot(~self_index_ordered & sorted_length>min_length,:);
    this_lengths = sorted_length(~self_index_ordered & sorted_length>min_length);
    matrix2images = matrix2plot;
    matrix2images(isnan(matrix2images))=0;   

    if ploting_bool
        subplot(2,3,4)
        imagesc(onset_time,1:numel(this_lengths),1-repmat(matrix2images,1,1,3));
        axis xy
        hold on
        plot((1:numel(this_lengths))*0, 1:numel(this_lengths), 'r')
        plot(this_lengths, 1:numel(this_lengths), 'r')
        plot((1:numel(this_lengths))*0 +pre_rate(1), 1:numel(this_lengths), ':r')
        xlim(ploting_range)
        yticks([])
        ylabel('Other')
    end

  
    rate_beofre_this_neuron = mean(matrix2plot(:,pre_rate_index ),2);
    call_before = mean(this_behaviort_aligned_calls_sorted(~self_index_ordered & sorted_length>min_length,pre_rate_index),2);  

    mdl1 = fitglm(call_before, this_lengths, 'Distribution', 'gamma', 'Link', 'log');
    rate_model = fitglm(rate_beofre_this_neuron, this_lengths, 'Distribution', 'gamma', 'Link', 'log');    
    glm_pvalue_2 = rate_model.coefTest;
    mdl2 = fitglm([call_before rate_beofre_this_neuron], this_lengths, 'Distribution', 'gamma', 'Link', 'log');
    D1 = mdl1.Deviance;  % simpler model
    D2 = mdl2.Deviance;  % full model

    % Get degrees of freedom difference
    df = mdl2.NumEstimatedCoefficients - mdl1.NumEstimatedCoefficients;

    % Compute test statistic
    chi2stat = D2 - D1;
    % Compute p-value using chi-squared distribution
    p_deviance2 = 1 - chi2cdf(chi2stat, df);

    % Estiamte correaltion beteen rate and length
    [c2,p2] = corr(rate_beofre_this_neuron,this_lengths, 'Type','Spearman');
     [c2_call,p2_call] = corr(call_before,this_lengths, 'Type','Spearman');
   
    if ploting_bool
        subplot(2,3,5)
        swarmchart(rate_beofre_this_neuron,this_lengths,'k.')
        title([c2 ,p2])

        subplot(2,3,6)
        swarmchart(call_before,this_lengths,'k.')
        title([c2_call ,p2_call])
        % saving_directory animal_code
        saveas(gcf,[saving_directory,  '\',this_neuron_area , ' ', animal_code,  ' length correlation ', this_neuron_area, ' ID' num2str(this_neuron), '.jpg'])
        pause(.1)
        close gcf
    end
    stacked_correlations(cn,1:10) = {c1 c2 p1 p2 p_deviance1 p_deviance2 glm_pvalue_1 glm_pvalue_2 p1_call p2_call };
    stacked_correlations(cn,11) = {this_neuron_area};
    stacked_correlations(cn,12) = {animal_code};
end

all_stacked_correlations = [all_stacked_correlations;stacked_correlations];
end


%%

all_stacked_correlations = cell2table(all_stacked_correlations);
all_stacked_correlations.Properties.VariableNames = {'CorrelationPouncing','CorrelationBeingPounce', 'PValPouncing','PValBeingPounce', 'DeviancePouncing','DevianceBeingPounce','GLMPvalue1','GLMPvalue2','PValPouncingCall','PValBeingPounceCall'   ,'Area', 'AnimalSession'};
all_stacked_correlations.Area(ismember(all_stacked_correlations.Area, 'isRT')) = {'isRt'};

%%
manual_order1 = [7 5 3 2 6 8 4 9 10 1];
manual_order2 = [5 3 1 4 6 2 7 8]; 
figure
subplot(1,2,1)
% significant = all_stacked_correlations.GLMPvalue1<0.05 & (all_stacked_correlations.DeviancePouncing<0.05 | all_stacked_correlations.PValPouncingCall>=0.05);
significant = all_stacked_correlations.GLMPvalue1<0.05;

sign_corr = all_stacked_correlations.CorrelationPouncing<0;


[counts_sig_neg, areas_sign_neg] = groupcounts(all_stacked_correlations.Area(significant & sign_corr));

sign_corr = all_stacked_correlations.CorrelationPouncing>0;

[counts_sig_pos, areas_sign_pos] = groupcounts(all_stacked_correlations.Area(significant & sign_corr));

[all_counts, all_areas] = groupcounts(all_stacked_correlations.Area);

bar_counts = zeros(numel(all_areas),4);

for j=1:numel(all_areas)
    loc = find(ismember(areas_sign_pos,all_areas{j}));
    if ~isempty(loc)
        bar_counts(j,1)  = counts_sig_pos(loc);
    end

    loc = find(ismember(areas_sign_neg,all_areas{j}));
     if ~isempty(loc)
        bar_counts(j,2)  = counts_sig_neg(loc);
     end
     bar_counts(j,3) =  all_counts(j)-sum(bar_counts(j,1:2));
     bar_counts(j,4) =  all_counts(j);
end

bar_proportions = diag(1./bar_counts(:,4))*bar_counts;

bar(100*bar_proportions(manual_order1, [2 1 3]), 'stacked')
xticks(1:numel(all_areas))
xticklabels(all_areas(manual_order1))
axis tight
ylim([0 20])


subplot(1,2,2)
significant = all_stacked_correlations.GLMPvalue1<0.05 & (all_stacked_correlations.DeviancePouncing<0.05 | all_stacked_correlations.PValPouncingCall>=0.05);
% significant = all_stacked_correlations.GLMPvalue1<0.05;

sign_corr = all_stacked_correlations.CorrelationPouncing<0;


[counts_sig_neg, areas_sign_neg] = groupcounts(all_stacked_correlations.Area(significant & sign_corr));

sign_corr = all_stacked_correlations.CorrelationPouncing>0;

[counts_sig_pos, areas_sign_pos] = groupcounts(all_stacked_correlations.Area(significant & sign_corr));

[all_counts, all_areas] = groupcounts(all_stacked_correlations.Area(all_stacked_correlations.PValPouncingCall>=0.05));

bar_counts = zeros(numel(all_areas),4);

for j=1:numel(all_areas)
    loc = find(ismember(areas_sign_pos,all_areas{j}));
    if ~isempty(loc)
        bar_counts(j,1)  = counts_sig_pos(loc);
    end

    loc = find(ismember(areas_sign_neg,all_areas{j}));
     if ~isempty(loc)
        bar_counts(j,2)  = counts_sig_neg(loc);
     end
     bar_counts(j,3) =  all_counts(j)-sum(bar_counts(j,1:2));
     bar_counts(j,4) =  all_counts(j);
end

bar_proportions = diag(1./bar_counts(:,4))*bar_counts;

bar(100*bar_proportions(manual_order2, [2 1 3]), 'stacked')
xticks(1:numel(all_areas))
xticklabels(all_areas(manual_order2))
axis tight
ylim([0 20])