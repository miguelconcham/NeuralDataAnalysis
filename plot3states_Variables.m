function [] = plot3states_Variables(variable_onset_struct, order)

ALL_VARIABLE_NAMES = variable_onset_struct.variable_types;
psth_edges = variable_onset_struct.psth_edges
filled_hmm_3states=variable_onset_struct.filled_play_bouts_3states;
beh_properties_onset_3states = variable_onset_struct.beh_properties_onset_3states;
beh_properties_offset_3states = variable_onset_struct.beh_properties_offset_3states;
[hmm_length_ordered, pb_order] = sort(diff(filled_hmm_3states(:,[2 3])'));
only_long = hmm_length_ordered'>=0
ordered_states = filled_hmm_3states(pb_order,1);

 order2show = 1:3;
  
for variable_n=1:numel(ALL_VARIABLE_NAMES)
    figure('units','normalized','outerposition',[0 0 1 1]);
   
    for hmm_n = 1:3
        index = ordered_states==(order2show(hmm_n)-1);
        colormap(1-gray)
        subplot(5,6,[1 7 13]+(order(hmm_n)-1)*2)
        matrix2plot = squeeze(beh_properties_onset_3states(order2show(hmm_n),variable_n,pb_order(only_long),:));
        matrix2plot = matrix2plot(index,:);
        imagesc(psth_edges,1:numel(hmm_length_ordered(only_long & index)), matrix2plot)
        hold on
        plot([0 0],[1 numel(hmm_length_ordered(only_long & index))],'r')
        hold on
        plot(hmm_length_ordered(only_long & index),1:numel(hmm_length_ordered(only_long & index)),'r')
        axis xy
        title(ALL_VARIABLE_NAMES{variable_n})

        subplot(5,6,[19 25]+(order(hmm_n)-1)*2)
        mean2plot = mean(matrix2plot);
        [~, ~, ci] = ttest(matrix2plot);

        fill([psth_edges fliplr(psth_edges )], [ci(1,:) fliplr(ci(2,:)) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none')
        hold on
        plot(psth_edges,mean2plot, 'k' )

        hold on
        plot([0 0],[min(min(ci)) max(max(ci))],'r')
        axis tight


        subplot(5,6,[1 7 13]+1+ (order(hmm_n)-1)*2)
        matrix2plot = squeeze(beh_properties_offset_3states(order2show(hmm_n),variable_n,pb_order(only_long),:));
        matrix2plot = matrix2plot(index,:);
        imagesc(psth_edges,1:numel(hmm_length_ordered(only_long & index)), matrix2plot)
        hold on
        plot([0 0],[1 numel(hmm_length_ordered(only_long & index))],'r')
        hold on
        plot(-hmm_length_ordered(only_long & index),1:numel(hmm_length_ordered(only_long & index)),'r')
        axis xy
        title(ALL_VARIABLE_NAMES{variable_n})

        subplot(5,6,[19 25]+1+(order(hmm_n)-1)*2)

        mean2plot = mean(matrix2plot);
        [~, ~, ci] = ttest(matrix2plot);

        fill([psth_edges fliplr(psth_edges )], [ci(1,:) fliplr(ci(2,:)) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none')
        hold on
        plot(psth_edges,mean2plot, 'k' )
        hold on
        plot([0 0],[min(min(ci)) max(max(ci))],'r')
        axis tight
    end
    pause(.1)
end