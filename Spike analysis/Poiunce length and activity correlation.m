pre_rate = [-.5 0];
pre_rate_index = onset_time>=pre_rate(1) & onset_time<=pre_rate(2);
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

cluster_index = 1:numel(cluster_area);

%%
stacked_correlations = [];
this_behaviort_aligned_calls_sorted = this_behaviort_aligned_calls(behavior_order,:);
% for cn = 1:numel(cluster_index)
for cn = 1:numel(cluster_index)
    figure('units','normalized','outerposition',[0 0 .75 1]);

    this_neuron = cluster_index(cn);
    this_neuron_area = cluster_area{this_neuron};

    

   subplot(2,3,1)
    % colormap(1-gray)
    matrix2plot = squeeze(psth_tensor_onset(this_neuron,behavior_order,:));
    matrix2plot = matrix2plot(self_index_ordered & sorted_length>min_length,:);
    this_lengths = sorted_length(self_index_ordered & sorted_length>min_length);
    matrix2images = matrix2plot;
    matrix2images(isnan(matrix2images))=0;   
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
        

    subplot(2,3,2)
    rate_beofre_this_neuron = mean(matrix2plot(:,pre_rate_index ),2);
    call_before = mean(this_behaviort_aligned_calls_sorted(self_index_ordered & sorted_length>min_length,pre_rate_index),2);  

    swarmchart(rate_beofre_this_neuron,this_lengths,'k.')

    [c1,p1] = corr(rate_beofre_this_neuron,this_lengths, 'Type','Spearman');
    title([c1 ,p1])




    subplot(2,3,3)

    swarmchart(call_before,this_lengths,'k.')

    [c,p] = corr(call_before,this_lengths, 'Type','Spearman');
    title([c ,p])



     subplot(2,3,4)
    % colormap(1-gray)
    matrix2plot = squeeze(psth_tensor_onset(this_neuron,behavior_order,:));
     matrix2plot = matrix2plot(~self_index_ordered & sorted_length>min_length,:);
    this_lengths = sorted_length(~self_index_ordered & sorted_length>min_length);
    matrix2images = matrix2plot;
    matrix2images(isnan(matrix2images))=0;   
    imagesc(onset_time,1:numel(this_lengths),1-repmat(matrix2images,1,1,3));
    axis xy
    hold on
    plot((1:numel(this_lengths))*0, 1:numel(this_lengths), 'r')
    plot(this_lengths, 1:numel(this_lengths), 'r') 
    plot((1:numel(this_lengths))*0 +pre_rate(1), 1:numel(this_lengths), ':r')
    xlim(ploting_range)
    yticks([])
    ylabel('Other')



    subplot(2,3,5)
    rate_beofre_this_neuron = mean(matrix2plot(:,pre_rate_index ),2);
    call_before = mean(this_behaviort_aligned_calls_sorted(~self_index_ordered & sorted_length>min_length,pre_rate_index),2);  

    swarmchart(rate_beofre_this_neuron,this_lengths,'k.')

    [c2,p2] = corr(rate_beofre_this_neuron,this_lengths, 'Type','Spearman');
    title([c2 ,p2])
    stacked_correlations = [stacked_correlations;[c1 c2 p1 p2]];

    subplot(2,3,6)

    swarmchart(call_before,this_lengths,'k.')

    [c,p] = corr(call_before,this_lengths, 'Type','Spearman');
    title([c ,p])

    saveas(gcf,['length correlation ', this_neuron_area, ' ID' num2str(this_neuron), '.jpg'])
    pause(.1)
    close gcf
end
% 


%%

figure
boxplot(stacked_correlations(:,1)-stacked_correlations(:,2),cluster_area )

