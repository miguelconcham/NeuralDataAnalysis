%% plot neurons
area2analyze      = 'PAG';
behaviors2analyse = {'Pounce_A','Pounce_Ai','Pounce_B','Pounce_Bi'};
% behaviors2analyse = {'CC','CB','CD','Pounce_A','Pounce_B','Evasion', 'Escape', 'Pin', 'Box'};
% behaviors2analyse = {'CC','CB','CD', 'Escape'};
current_dir = cd;
ego = 0;
range_borders = [-2 2];
bin_size = 0.01;
 [psth_tensor_onset, psth_tensor_offset,psth_edges_onset,psth_edges_offset, mean_rate_during_behavior, behavior_indexes, cluster_area, beh_length, Behavior,ego_alter_dummy, self_animal,cluster_info] =get_psth(current_dir,behaviors2analyse,area2analyze,range_borders,bin_size);

%%
 rgbColors = generateDistinctColors(numel(behaviors2analyse));

figure;
hold on;
n = numel(behaviors2analyse)
for i = 1:n
    plot([1 10], i*[1 1], 'Color', rgbColors(i,:), 'LineWidth', 2);
end
hold off;
legend(behaviors2analyse)

play_beahvior_index = ~ismember(Behavior.Type, {'Partners session','Tickling'});
ego_index           = ismember(Behavior.Animal, self_animal);
%%
min_length = 0.25;
ploting_range = [-1 2];
[sorted_length, behavior_order] = sort(beh_length);
Behavior_onset_offset = [Behavior.Start(behavior_indexes) Behavior.End(behavior_indexes)];
Behavior_type = Behavior.Type(behavior_indexes);
self_index_ordered = ego_alter_dummy(behavior_order);

cluster_index = find(ismember(cluster_area, {'DLPAG','LPAG','VLPAG', 'SupCol', 'DR','isRt'}));
onset_time = .5*(psth_edges_onset(1:end-1)+psth_edges_onset(2:end));
staked_responses    = nan(numel(cluster_index),2*numel(behaviors2analyse) +1, numel(onset_time));
staked_correlations = nan(numel(cluster_index),2*numel(behaviors2analyse) +1, 2*numel(behaviors2analyse) +1);
staked_p_values     = nan(numel(cluster_index),2*numel(behaviors2analyse) +1, 2*numel(behaviors2analyse) +1);
staked_call_correlations = nan(numel(cluster_index),2*numel(behaviors2analyse),2);
staked_curve_distances = nan(numel(cluster_index),2*numel(behaviors2analyse) + 2,2);
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

this_behavior_aligned_beheavior = nan(size(Behavior_onset_offset,1),numel(psth_edges_onset)-1);

for bn = 1:size(Behavior_onset_offset,1)
    this_behavior_onset  = Behavior_onset_offset(bn,1);
    this_behavior_offset = Behavior_onset_offset(bn,2);

    this_behavior_aligned_beheavior(bn,:) = any((psth_edges_onset(1:end-1)+this_behavior_onset>=Behavior.Start(play_beahvior_index) & psth_edges_onset(1:end-1)+this_behavior_onset<=Behavior.End(play_beahvior_index)) | ...
        (psth_edges_onset(2:end)+this_behavior_onset>=Behavior.Start(play_beahvior_index) & psth_edges_onset(2:end)+this_behavior_onset<=Behavior.End(play_beahvior_index)),1);
    this_behavior_aligned_beheavior(bn,psth_edges_onset(1:end-1)>range_borders(2)+this_behavior_offset-this_behavior_onset)  = 0;
end

self_aligned_behavior  = nan(size(Behavior_onset_offset,1),numel(psth_edges_onset)-1);
other_aligned_behavior = nan(size(Behavior_onset_offset,1),numel(psth_edges_onset)-1);

for bn = 1:size(Behavior_onset_offset,1)
    this_behavior_onset  = Behavior_onset_offset(bn,1);
    this_behavior_offset = Behavior_onset_offset(bn,2);

    self_aligned_behavior(bn,:) ...
        = any((psth_edges_onset(1:end-1)+this_behavior_onset>=Behavior.Start(ego_index) & psth_edges_onset(1:end-1)+this_behavior_onset<=Behavior.End(ego_index)) | ...
              (psth_edges_onset(2:end)+this_behavior_onset>=Behavior.Start(ego_index) & psth_edges_onset(2:end)+this_behavior_onset<=Behavior.End(ego_index)),1);
    self_aligned_behavior(bn,psth_edges_onset(1:end-1)>range_borders(2)+this_behavior_offset-this_behavior_onset)  = 0;


    other_aligned_behavior(bn,:) ...
        = any((psth_edges_onset(1:end-1)+this_behavior_onset>=Behavior.Start(~ego_index) & psth_edges_onset(1:end-1)+this_behavior_onset<=Behavior.End(~ego_index)) | ...
              (psth_edges_onset(2:end)+this_behavior_onset>=Behavior.Start(~ego_index) & psth_edges_onset(2:end)+this_behavior_onset<=Behavior.End(~ego_index)),1);
    other_aligned_behavior(bn,psth_edges_onset(1:end-1)>range_borders(2)+this_behavior_offset-this_behavior_onset)  = 0;
end



for cn = 1:numel(cluster_index)
    this_neuron = cluster_index(cn);
    this_neuron_area = cluster_area{this_neuron};

    

    figure('units','normalized','outerposition',[0 0 .75 1]);
    subplot(4,2,1)
    % colormap(1-gray)
    matrix2plot = squeeze(psth_tensor_onset(this_neuron,behavior_order,:));
    matrix2images = matrix2plot;
    matrix2images(isnan(matrix2images))=0;
    matrix2images = matrix2images(self_index_ordered & sorted_length>min_length,:);
    this_lengths = sorted_length(self_index_ordered & sorted_length>min_length);
    imagesc(onset_time,1:numel(this_lengths),1-repmat(matrix2images,1,1,3));
    axis xy
    hold on
    plot((1:numel(this_lengths))*0, 1:numel(this_lengths), 'r')
    plot(this_lengths, 1:numel(this_lengths), 'r') 
    xlim(ploting_range)
    yticks([])
     xticklabels([])
    ylabel('Self')
    title(['Behavior ', this_neuron_area, ' ID' num2str(this_neuron)])


     subplot(4,2,3)
    % colormap(1-gray)
    matrix2plot = squeeze(psth_tensor_onset(this_neuron,behavior_order,:));
    matrix2images = matrix2plot;
    matrix2images(isnan(matrix2images))=0;
    matrix2images = matrix2images(~self_index_ordered & sorted_length>min_length,:);
    imagesc(onset_time,1:numel(this_lengths),1-repmat(matrix2images,1,1,3));
    axis xy
    hold on
    plot((1:numel(this_lengths))*0, 1:numel(this_lengths), 'r')
    plot(this_lengths, 1:numel(this_lengths), 'r') 
    xlim(ploting_range)
    yticks([])
     xticklabels([])
    ylabel('Other')





    subplot(5,2,2)
    hold on
    long_enough_behavior = behavior_order( sorted_length>min_length);
     for bn=1:numel(long_enough_behavior)
        this_behavior_start = Behavior_onset_offset(long_enough_behavior(bn),1);
        behaviors_around = find((Behavior.End>=this_behavior_start+ploting_range(1) & Behavior.End<=this_behavior_start+ploting_range(2)) | ...
            (Behavior.Start>=this_behavior_start+ploting_range(1) & Behavior.Start<=this_behavior_start+ploting_range(2)));
        for sub_bn = 1:numel(behaviors_around)
            sub_behavior_start = Behavior.Start(behaviors_around(sub_bn));
            sub_behavior_end   = Behavior.End(behaviors_around(sub_bn));
            this_behavior_type = Behavior.Type(behaviors_around(sub_bn));
            color_type = find(ismember(behaviors2analyse,this_behavior_type));

            if ~isempty(color_type)                      
                fill([sub_behavior_start sub_behavior_end sub_behavior_end sub_behavior_start]-this_behavior_start, [-.5 -.5 .5 .5]+bn, 'k', 'FaceColor', rgbColors(color_type,:), 'EdgeColor', 'none', 'FaceAlpha',.5)
            else
                  fill([sub_behavior_start sub_behavior_end sub_behavior_end sub_behavior_start]-this_behavior_start, [-.5 -.5 .5 .5]+bn,'k',  'EdgeColor', 'none', 'FaceAlpha',.1)
            end
        end
     end
     xlim(ploting_range)
     xticklabels ([])
     ylim([.5 numel(long_enough_behavior)+.5])
     yticks([])
     ylabel('Behvior Type')


    subplot(5,2,4)
     hold on
     for bn=1:numel(long_enough_behavior)
        this_behavior_start = Behavior_onset_offset(long_enough_behavior(bn),1);
        behaviors_around = find((Behavior.End>=this_behavior_start+ploting_range(1) & Behavior.End<=this_behavior_start+ploting_range(2)) | ...
            (Behavior.Start>=this_behavior_start+ploting_range(1) & Behavior.Start<=this_behavior_start+ploting_range(2)));
        for sub_bn = 1:numel(behaviors_around)
            sub_behavior_start = Behavior.Start(behaviors_around(sub_bn));
            sub_behavior_end   = Behavior.End(behaviors_around(sub_bn));
            this_behavior_animal = Behavior.Animal(behaviors_around(sub_bn));
            color_type = ismember(this_behavior_animal,self_animal);

            if ~color_type                      
                fill([sub_behavior_start sub_behavior_end sub_behavior_end sub_behavior_start]-this_behavior_start, [-.5 -.5 .5 .5]+bn, 'r', 'EdgeColor', 'none', 'FaceAlpha',.25)
            else
                  fill([sub_behavior_start sub_behavior_end sub_behavior_end sub_behavior_start]-this_behavior_start, [-.5 -.5 .5 .5]+bn,'b',  'EdgeColor', 'none', 'FaceAlpha',.25)
            end
        end
     end
     xlim(ploting_range)
     ylim([.5 numel(long_enough_behavior)+.5])
     yticks([])
    ylabel('Self/Other')
    xticklabels ([])


    subplot(5,2,6)
    hold off

     matrix2images = this_behaviort_aligned_calls(behavior_order(sorted_length>min_length),:);
    matrix2images(isnan(matrix2images))=0;

    imagesc(onset_time,1:size(matrix2images,1),1-repmat(matrix2images,1,1,3));
    axis xy
    axis xy
    hold on
    plot((1:sum(sorted_length>min_length))*0, 1:sum(sorted_length>min_length), 'r')
    plot(sorted_length(sorted_length>min_length), 1:sum(sorted_length>min_length), 'r')
    xlim(ploting_range)
      yticks([])
    ylabel('Join USV')
    xlabel('Time (s)')

% 
% if exist('CallStats', 'var')
% 
%     this_behaviort_aligned_calls = nan(size(Behavior_onset_offset,1),numel(psth_edges_onset)-1);
%     for bn = 1:size(Behavior_onset_offset,1)
%         this_behavior_onset = Behavior_onset_offset(bn,1);
%          calls_around = find(CallStats.EndTimes-this_behavior_onset>=ploting_range(1) & CallStats.BeginTimes<=this_behavior_onset+ploting_range(2));
%         calls_around_behavior = (psth_edges_onset(1)+this_behavior_onset>=CallStats.BeginTimes & psth_edges_onset(1:end-1)+this_behavior_onset<=CallStats.EndTimes)
% 
%     end
% end


matrix2plot = squeeze(psth_tensor_onset(this_neuron,:,:));

subplot(4,2,5)
yyaxis right
hold on
plot(onset_time,zscore(movmean(mean(this_behaviort_aligned_calls(ego_alter_dummy & beh_length>min_length,:), 'omitmissing'),10)), 'r')
plot(onset_time,zscore(movmean(mean(this_behavior_aligned_beheavior(ego_alter_dummy & beh_length>min_length,:), 'omitmissing'),10)), 'b')
plot(onset_time,zscore(movmean(mean(other_aligned_behavior(ego_alter_dummy & beh_length>min_length,:), 'omitmissing'),10)), ':m', 'LineWidth',2)
xlim(ploting_range)
ylim tight
ylabel('z score')
yyaxis left
plot(onset_time,movmean(mean(matrix2plot(ego_alter_dummy & (beh_length>min_length),:), 'omitmissing'),10)/bin_size, 'k', 'LineWidth',2)
hold on
ylabel({'Self','Rate'})
hold on
y_lim = ylim;
plot([0 0], y_lim, 'r')
xlim(ploting_range)

%exctarct curve similarity

y1 = movmean(mean(this_behaviort_aligned_calls(ego_alter_dummy & beh_length>min_length,:), 'omitmissing'),10);
y2 = movmean(mean(matrix2plot(ego_alter_dummy & (beh_length>min_length),:), 'omitmissing'),10)/bin_size;
no_nan = ~isnan(y1) & ~isnan(y2) & onset_time<range_borders(2) & onset_time>range_borders(1);
y1= zscore(y1(no_nan));
y2= zscore(y2(no_nan));

curve_Distance1 = sqrt(sum((y1-y2).^2))/numel(y1);
curve_Distance2  = sum(y1.*y2)/(norm(y1)*norm(y2));
staked_curve_distances(cn,2*numel(behaviors2analyse)+1,1) = curve_Distance1;
staked_curve_distances(cn,2*numel(behaviors2analyse)+1,2) = curve_Distance2;





subplot(4,2,7)
yyaxis right
hold on
plot(onset_time,zscore(movmean(mean(this_behaviort_aligned_calls(~ego_alter_dummy & beh_length>min_length,:), 'omitmissing'),10)), 'r')
plot(onset_time,zscore(movmean(mean(this_behavior_aligned_beheavior(~ego_alter_dummy & beh_length>min_length,:), 'omitmissing'),10)), 'b')
plot(onset_time,zscore(movmean(mean(other_aligned_behavior(~ego_alter_dummy & beh_length>min_length,:), 'omitmissing'),10)), ':m', 'LineWidth',2)
xlim(ploting_range)
ylim tight
ylabel('z score')
yyaxis left
plot(onset_time,movmean(mean(matrix2plot(~ego_alter_dummy & (beh_length>min_length),:), 'omitmissing'),10)/bin_size, 'k', 'LineWidth',2)
hold on
ylabel({'Other','Rate'})
hold on
y_lim = ylim;
plot([0 0], y_lim, 'r')
xlim(ploting_range)
xlabel('Time (s)')


%exctarct curve similarity
y1 = movmean(mean(this_behaviort_aligned_calls(~ego_alter_dummy & beh_length>min_length,:), 'omitmissing'),10);
y2 = movmean(mean(matrix2plot(~ego_alter_dummy & (beh_length>min_length),:), 'omitmissing'),10)/bin_size;
no_nan = ~isnan(y1) & ~isnan(y2) & onset_time<range_borders(2) & onset_time>range_borders(1);
y1= zscore(y1(no_nan));
y2= zscore(y2(no_nan));
curve_Distance1 = sqrt(sum((y1-y2).^2))/numel(y1);
curve_Distance2  = sum(y1.*y2)/(norm(y1)*norm(y2));
staked_curve_distances(cn,2*numel(behaviors2analyse)+2,1) = curve_Distance1;
staked_curve_distances(cn,2*numel(behaviors2analyse)+2,2) = curve_Distance2;





matrix2plot = squeeze(psth_tensor_onset(this_neuron,:,:));
subplot(5,2,8)
staked_labels = [];
animal_label = {'Other','Self'};
onset_ordering_time = onset_time>=0 & onset_time<=min_length;
marginall_call   = nan(2*numel(behaviors2analyse) +1,numel(onset_time));
staked_behavior  = nan(2*numel(behaviors2analyse) +1,numel(onset_time));
col = 1;
for so =0:1
    for bn = 1:numel(behaviors2analyse)
        behaviortype = ismember(Behavior_type, behaviors2analyse{bn});
        animaltype   = ego_alter_dummy==so;
        
        if sum(behaviortype & (beh_length>min_length) & animaltype)>1
            mean_value = movmean(mean(matrix2plot(behaviortype & (beh_length>min_length) & animaltype,:), 'omitmissing'),10);
            mean_value =( mean_value - mean(mean_value, 'omitmissing'))/std(mean_value, 'omitmissing');
            staked_behavior(col,:) = mean_value;
            marginall_call(col,:) =  movmean(mean(this_behaviort_aligned_calls(behaviortype & (beh_length>min_length) & animaltype,:), 'omitmissing'),10);
        elseif sum(behaviortype & (beh_length>beh_length) & animaltype)==1
             mean_value = movmean(matrix2plot(behaviortype & (beh_length>min_length) & animaltype,:),10);
             mean_value =( mean_value - mean(mean_value, 'omitmissing'))/std(mean_value, 'omitmissing');
            staked_behavior(col,:) = mean_value;
            marginall_call(col,:) =  movmean((this_behaviort_aligned_calls(behaviortype & (beh_length>min_length) & animaltype,:)),10);
        else
           
        end



        staked_labels = [staked_labels,{[animal_label{so+1},' ',behaviors2analyse{bn}]}];
        col = col+1;
    end
end



staked_behavior(col,:) = [zscore(movmean(mean(this_behaviort_aligned_calls(beh_length>min_length,:), 'omitmissing'),10))];

staked_responses(cn,:,:) = staked_behavior;

staked_labels = [staked_labels, {'Join USV'}];

onset_response = mean(staked_behavior(:,onset_ordering_time),2);
[~, order] = sort(onset_response);
imagesc(onset_time, 1:(numel(behaviors2analyse)*2 + 1),staked_behavior(order,:))
axis xy
yticks(1:(2*numel(behaviors2analyse)+1))
yticklabels(staked_labels(order))
% 
% for bn = 1:numel(behaviors2analyse)
%     datatype = ismember(Behavior_type, behaviors2analyse{bn});
%     plot(onset_time,movmean(mean(matrix2plot(datatype,:), 'omitmissing'),10), 'Color', rgbColors(bn,:))
%     hold on
% end
hold on
y_lim = ylim;

plot([0 0], y_lim, 'w')
plot([.5 .5], y_lim, ':w')
xlim(ploting_range)
ylim(y_lim)

[corr_matrix, p] = corr(staked_behavior');

for j=1:size(corr_matrix,1)
    for i=1:size(corr_matrix,2)
        no_nan = ~isnan(staked_behavior(j,:)) & ~isnan(staked_behavior(i,:));
        if sum(no_nan)>3
            [corr_matrix(j,i), p(i,j)] = corr(staked_behavior(j,no_nan)',staked_behavior(i,no_nan)');
        end
    end
end
call_correaltion = nan(2*numel(behaviors2analyse),2);
for j=1:size(call_correaltion,1)
    no_nan = ~isnan(staked_behavior(j,:)) & ~isnan(marginall_call(j,:)) & onset_time<range_borders(2) & onset_time>range_borders(1);
    if sum(no_nan)>3
        y1 = zscore(staked_behavior(j,no_nan))';
        y2 = zscore(marginall_call(j,no_nan))';
   
        [call_correaltion(j,1), call_correaltion(j,2)] = corr(y1,y2);
        curve_Distance1 = sqrt(sum((y1-y2).^2))/numel(y1);
        curve_Distance2  = sum(y1.*y2)/(norm(y1)*norm(y2));
        staked_curve_distances(cn,j,1) = curve_Distance1;
        staked_curve_distances(cn,j,2) = curve_Distance2;
    end
        
end



staked_correlations(cn,:,:) =corr_matrix;
staked_p_values(cn,:,:) = p;
staked_call_correlations(cn,:,:) = call_correaltion;

subplot(5,4,19)
imagesc(corr_matrix(order,order))
axis xy
xticks(1:(2*numel(behaviors2analyse)+1))
xticklabels(staked_labels(order))

yticks(1:(2*numel(behaviors2analyse)+1))
yticklabels(staked_labels(order))
clim([-.75 .75])

subplot(5,4,20)
var2plot = 2;
this_stacked_distances = squeeze(staked_curve_distances(cn,:,:));
hold on
plot(repmat(1:size(this_stacked_distances,1),2,1),[this_stacked_distances(:,1)*0,this_stacked_distances(:,var2plot)]', 'k')
plot(repmat(1:size(this_stacked_distances,1),2,1),[this_stacked_distances(:,1)*0,this_stacked_distances(:,var2plot)]', '.k')
x_pos = 1:size(this_stacked_distances,1);
xticks(x_pos)
xticklabels([staked_labels(1:end-1), 'Self', 'Other'])
ylim([-1 1])
ylabel('Call-Response Similarty')
% plot(repmat(1:size(call_correaltion,1),2,1),[call_correaltion(:,1)*0,call_correaltion(:,1)]', 'k')
% plot(repmat(1:size(call_correaltion,1),2,1),[call_correaltion(:,1)*0,call_correaltion(:,1)]', 'k')
% plot(1:size(call_correaltion,1),call_correaltion(:,1), '.k', 'MarkerSize',5)
% hold on
% significant = call_correaltion(:,2)<0.01;
% x_pos = 1:size(call_correaltion,1);
% plot(x_pos,call_correaltion(:,1), '.k', 'MarkerSize',5)
% plot(x_pos(significant),call_correaltion((significant),1), '.r', 'MarkerSize',10)
% xticks(1:size(call_correaltion,1))
% xlim([.5 size(call_correaltion,1)+.5])
% xticklabels(staked_labels(1:end-1))
% ylim([-1 1])
% ylabel('r')




pause(.1)
saveas(gcf, ['Behavior ', this_neuron_area, ' ID' num2str(this_neuron), '.jpg'])
close gcf
end

%% saving result
summary_result = [];
summary_result.cluster_index            = cluster_index;
summary_result.cluster_info             = cluster_info;
summary_result.cluster_area             = cluster_area;
summary_result.staked_responses         = staked_responses;
summary_result.staked_call_correlations = staked_call_correlations;
summary_result.staked_correlations      = staked_correlations;
summary_result.staked_p_values          = staked_p_values;
summary_result.staked_labels            = staked_labels;
summary_result.onset_time               = onset_time;
summary_result.staked_curve_distances   = staked_curve_distances;
save('summary_result','summary_result')

%%
 
load summary_result.mat
extractStructFields(summary_result)

[~,depth_order] = sort(cluster_info.depth(cluster_index), 'ascend');
area_name = cluster_area(cluster_index);
area_name = area_name(depth_order);

 nameCell = area_name(:);  
    
    % Find change points
    changeIdx = [1; find(~strcmp(nameCell(1:end-1), nameCell(2:end))) + 1];
    endIdx = [changeIdx(2:end)-1; length(nameCell)];
    
    % Extract names
    clusterNames = nameCell(changeIdx);
    
    % Create table
    nameTable = table(changeIdx, endIdx, clusterNames, ...
                         'VariableNames', {'StartIdx', 'EndIdx', 'Name'});

figure
for j=1:size(staked_responses,2)-1
    subplot(4,5,j)
    imagesc(onset_time, 1:numel(depth_order),squeeze(staked_responses(depth_order,j,:)))
    axis xy
    hold on
    for k=1:size(nameTable,1)
        plot([onset_time(1) onset_time(end)],[1 1]*nameTable.StartIdx(k),'w', 'LineWidth',2)
        plot([onset_time(1) onset_time(end)],[1 1]*nameTable.EndIdx(k)+1,'w', 'LineWidth',2)
    end
    hold on
    plot([0 0],[ 1 numel(depth_order)], 'r')
    yticks(.5*(nameTable.StartIdx + nameTable.EndIdx))
    yticklabels(nameTable.Name)
    title(staked_labels{j})
    clim([-2 2])
end

%%
area_name = cluster_area(cluster_index);
areas2plot = {'VLPAG','LPAG','DLPAG','SupCol'}
colors = {'r','b','k','m'}
figure('units','normalized','outerposition',[0 0 .75 1]);
sp = 1;
for j=1:size(staked_responses,2)-1
    if sp>4
        figure('units','normalized','outerposition',[0 0 .75 1]);
        sp=1;
    end
    subplot(2,2,sp)
    for an = 1:numel(areas2plot)
        matrix2plot = squeeze(staked_responses(ismember(area_name,areas2plot(an)),j,:));
        promedio =  mean(matrix2plot, 'omitmissing');
        promedio = movmean(promedio, 10);
        plot(onset_time,promedio, 'Color', colors{an})        
        hold on    
    end
    y_lim = ylim;
    plot([0 0], y_lim, 'k', 'HandleVisibility','off')
    axis tight
    legend(areas2plot)
    title(staked_labels{j})
        sp=sp+1;
end
% staked_responses
% staked_correlations(cn,:,:) =corr_matrix;
% staked_p_values(cn,:,:) = p;
% staked_call_correlations(cn,:,:) = call_correaltion;

%%

[sorted_length, behavior_order] = sort(beh_length);
area_list = unique(cluster_area);


for an= 1:numel(area_list)

cluster_index = find(ismember(cluster_area, area_list(an)));
onset_time = .5*(psth_edges_onset(1:end-1)+psth_edges_onset(2:end));
% 
% zscored_data = squeeze(psth_tensor_onset(cluster_index,behavior_order,:));
% for nn = 1:size(zscored_data,1)
%     this_neuron = squeeze(zscored_data(nn,:,:));
%     this_neuron(~isnan(this_neuron)) = zscore(  this_neuron(~isnan(this_neuron)));
%     zscored_data(nn,:,:) = this_neuron;
% end


matrix2plot = squeeze(sum(psth_tensor_onset(cluster_index,behavior_order,:)));      
 % matrix2plot = squeeze(mean(zscored_data));

for j=1:size(matrix2plot,1)
    matrix2plot(j,:) = movmean(matrix2plot(j,:),10);
end
        

   

figure
subplot(3,1,1:2)
colormap(1-gray)
imagesc(onset_time,1:numel(behavior_order),matrix2plot);
axis xy
hold on
plot((1:numel(behavior_order))*0, 1:numel(behavior_order), 'r')
plot(sorted_length, 1:numel(behavior_order), 'r')


title(area_list(an))


subplot(3,1,3)
if numel(cluster_index)>1
    mean_rate = mean(mean_rate_during_behavior(cluster_index,:,1)-mean_rate_during_behavior(cluster_index,:,2));
else
    mean_rate = (mean_rate_during_behavior(cluster_index,:,1)-mean_rate_during_behavior(cluster_index,:,2));
end

[~,legth_out] = rmoutliers(beh_length');
[~,rate_out] = rmoutliers(mean_rate);
data2plot = ~isnan(legth_out) & ~isnan(rate_out);
plot(beh_length(data2plot),mean_rate(data2plot), '.')
[c,p] =corr(beh_length(data2plot),mean_rate(data2plot)', 'Type','Spearman');
title([num2str(c), ' ', num2str(p)])

end





