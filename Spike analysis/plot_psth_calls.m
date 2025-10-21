area2analyze      = 'PAG';
behaviors2analyse = {'Pounce_A','Pounce_Ai','Pounce_B','Pounce_Bi'};
behaviors2analyse = {'CC','CB','CD','Pounce_A','Pounce_B','Evasion', 'Escape', 'Pin', 'Box'};
current_dir = cd;
range_borders = [-2 2];
bin_size = 0.01;
[psth_tensor_onset_call, psth_tensor_offset_call,psth_edges_onset_call,psth_edges_offset_call, mean_rate_during_call, all_call_index, cluster_area, call_length, CallStats, call_type,calls_within,psth_calls_onset,behavior_type] = get_psth_calls(current_dir,behaviors2analyse,area2analyze,range_borders,bin_size);
%%
ploting_range = [-.25 .25];

figure
[sorted_length, call_order] = sort(call_length);
ordeded_call_index = all_call_index(call_order);
condition2plot = {'Thikling', 'DuringBehavior','PlaySession'};
data2include = ismember(call_type(call_order), 'Thikling');
% data2include = ismember(call_type(call_order), 'DuringBehavior');

area2plot = {'VLPAG', 'DR', 'SupCol', 'LPAG'};
cluster_index = find(ismember(cluster_area, area2plot));
onset_time = .5*(psth_edges_onset_call(1:end-1)+psth_edges_onset_call(2:end));
this_call_lengthts = call_length(data2include);


for cn = 1:numel(cluster_index)


    this_neuron = cluster_index(cn);
    this_neuron_area = cluster_area(cluster_index(cn));
   

    figure('units','normalized','outerposition',[0 0 1 1]);

    for din = 1:numel(condition2plot)
        data2include = ismember(call_type(call_order), condition2plot{din});
        matrix2plot = squeeze(psth_tensor_onset_call(this_neuron,call_order(data2include),:));
        calls_this_condition = ordeded_call_index(data2include);

        colormap(1-gray)
        subplot(3,numel(condition2plot),(1:numel(condition2plot):2*numel(condition2plot))  + din-1)
        imagesc(onset_time,1:numel(call_order(data2include)),matrix2plot);
        axis xy
        hold on
        plot((1:numel(call_order(data2include)))*0, 1:numel(call_order(data2include)), 'r')
        plot(sorted_length(data2include), 1:numel(call_order(data2include)), 'r')
        clim([0 1])


        for calln=1:numel(calls_this_condition)

            call_start = CallStats.BeginTimes(calls_this_condition(calln));
            calls_around = find(CallStats.EndTimes-call_start>=ploting_range(1) & CallStats.BeginTimes<=call_start+ploting_range(2));

            for sub_cn = 1:numel(calls_around)
                call_around_start = CallStats.BeginTimes(calls_around(sub_cn))-call_start;
                call_around_end   = CallStats.EndTimes(calls_around(sub_cn))-call_start;
                fill([call_around_start call_around_end  call_around_end call_around_start], [-.5 -.5 .5 .5]+calln, 'r', 'FaceAlpha',.25, 'EdgeColor','none')
            end
        end
        xlim(ploting_range)
           title([condition2plot{din}, ' ', this_neuron_area{1},  ' ID', num2str(this_neuron)])
        subplot(3,numel(condition2plot),2*numel(condition2plot)  +  din)
        plot(onset_time, movmean(mean(matrix2plot),5))
        hold on
        y_lim = ylim;
        plot([0 0], y_lim, 'r')
        xlim(ploting_range)
        pause(.1)
    end
end


%%

figHandles = findall(0, 'Type', 'figure');

% Loop over each figure and save it as a JPG
for i = 1:length(figHandles)
    fig = figHandles(i);
    % Create a filename using the figure number
    filename = sprintf('Figure_%d.jpg', fig.Number);
    % Save the figure as a JPG in the current folder
    saveas(fig, filename);
    close(fig)
end


%%
selcted_cells = ismember(cluster_index,[53 52 51 48 47]);

figure

for din = 1:numel(condition2plot)
    data2include = ismember(call_type(call_order), condition2plot{din});
    matrix2plot = squeeze(sum(psth_tensor_onset_call(cluster_index(selcted_cells),call_order(data2include),:)));

    colormap(1-gray)
    subplot(3,numel(condition2plot),(1:numel(condition2plot):2*numel(condition2plot))  + din-1)
    imagesc(onset_time,1:numel(call_order(data2include)),matrix2plot);
    axis xy
    hold on
    plot((1:numel(call_order(data2include)))*0, 1:numel(call_order(data2include)), 'r')
    plot(sorted_length(data2include), 1:numel(call_order(data2include)), 'r')
    title([condition2plot{din}, ' ', this_neuron_area{1},  ' ID', num2str(this_neuron)])
    xlim([-.5 .5])

    subplot(3,numel(condition2plot),2*numel(condition2plot)  +  din)
    plot(onset_time, movmean(mean(matrix2plot),5))
    hold on
    y_lim = ylim;
    plot([0 0], y_lim, 'r')
    xlim([-.5 .5])
    pause(.1)
    end

%%



figure

data2include = ismember(call_type, 'DuringBehavior');
call_indexes = all_call_index(data2include);
selected_call_length = call_length(data2include);
behavior_length = calls_within(ismember(calls_within(:,1),call_indexes),:);

[~, index_order] = sort(behavior_length(:,1));

behavior_length =behavior_length(index_order,:);
behavior_length(diff(behavior_length(:,1))==0,:) = [];
[~, index_order] = sort(behavior_length(:,1));






cluster_index = find(ismember(cluster_area, 'LPAG'));
onset_time = .5*(psth_edges_onset_call(1:end-1)+psth_edges_onset_call(2:end));
[behavior_length, pounce_order] = sort(behavior_length(:,2));
for cn = 1:numel(cluster_index)
    this_neuron = cluster_index(cn);
    matrix2plot = squeeze(psth_tensor_onset_call(this_neuron,data2include,:));
    matrix2plot = matrix2plot(pounce_order,:);
    figure('units','normalized','outerposition',[.5 0 .25 1]);
    colormap(1-gray)
    subplot(3,1,[1 2])
    imagesc(onset_time,1:size(matrix2plot,1),matrix2plot);
    axis xy
    hold on
    plot((1:size(matrix2plot,1))*0, 1:size(matrix2plot,1), 'r')
    plot(selected_call_length, 1:numel(selected_call_length), 'r')
    title(['Call ', num2str(this_neuron)])
    xlim([-.5 .5])

    subplot(3,1,3)
    plot(onset_time, movmean(mean(matrix2plot),5))
    hold on
    y_lim = ylim;
    plot([0 0], y_lim, 'r')
    xlim([-.5 .5])
    pause(.1)
end

%%

selcted_cells = ismember(cluster_index,[53 52 51 48 47]);
this_call_lengthts = call_length(data2include);
dataindexes = find(data2include);
this_call_lengthts = call_length(data2include);
[~, call_reorder] = sort(this_call_lengthts);
dataindexes = dataindexes(call_reorder);
[~, call_reorder] = sort(this_call_lengthts);

 matrix2plot = squeeze(sum(psth_tensor_onset_call(cluster_index(selcted_cells),data2include,:)));
    matrix2plot = matrix2plot(index_order,:);
    figure('units','normalized','outerposition',[.5 0 .25 1]);
    colormap(1-gray)
    subplot(3,1,[1 2])
    imagesc(onset_time,1:size(matrix2plot,1),matrix2plot);
    axis xy
    hold on
    plot((1:size(matrix2plot,1))*0, 1:size(matrix2plot,1), 'r')
    plot(selected_call_length(index_order), 1:numel(selected_call_length(index_order)), 'r')
    title(['Call ', num2str(this_neuron)])
    xlim([-.5 .5])

    subplot(3,1,3)
    plot(onset_time, movmean(mean(matrix2plot),5), 'k')
    hold on
    y_lim = ylim;
    plot([0 0], y_lim, 'r')
    xlim([-.5 .5])
    ylim(y_lim)
    pause(.1)