dir_list = dir('*.mat');



shortest_time = Inf;
for fn=1:numel(dir_list)

  load(dir_list(fn).name)
  time_size = numel(summary_result.onset_time);
shortest_time = min(shortest_time, time_size);
disp(shortest_time)
end


fn =1;
load(dir_list(fn).name)

all_summary_result = summary_result;

if ~any(ismember(fieldnames(summary_result), 'staked_curve_distances'))

    summary_result.staked_curve_distances = nan(size(summary_result.staked_correlations,1),size(summary_result.staked_correlations,2)+1,2);
end

field_names = fieldnames(summary_result);
field_names(ismember(field_names, {'onset_time','staked_labels'})) = [];
for j=1:numel(field_names)
    if ismember(field_names{j},'staked_responses' )
        all_summary_result.(field_names{j}) = summary_result.(field_names{j})(:,:, 1:shortest_time);
    elseif ismember(field_names{j},'cluster_area' )
        all_summary_result.cluster_area = summary_result.cluster_area(summary_result.cluster_index);
    elseif ismember(field_names{j},'cluster_info' )
        all_summary_result.cluster_info = summary_result.cluster_info(summary_result.cluster_index,:);
    else

        all_summary_result.(field_names{j}) = summary_result.(field_names{j});
    end
end

all_summary_result.cluster_info.Session = ones(size(all_summary_result.cluster_info,1),1)*fn;
clear summary_result

%%

for fn=2:numel(dir_list)

    load(dir_list(fn).name)

    if ~any(ismember(fieldnames(summary_result), 'staked_curve_distances'))

        summary_result.staked_curve_distances = nan(size(summary_result.staked_correlations,1),size(summary_result.staked_correlations,2)+1,2);
    end
    summary_result.cluster_info.Session = ones(size(summary_result.cluster_info,1),1)*fn;
    field_names = fieldnames(summary_result);
    field_names(ismember(field_names, {'onset_time','staked_labels'})) = [];
    for j=1:numel(field_names)
        if ismember(field_names{j},'staked_responses' )
            all_summary_result.(field_names{j}) = cat(1,all_summary_result.(field_names{j}),summary_result.(field_names{j})(:,:, 1:shortest_time));
        elseif ismember(field_names{j},'cluster_area' )
            all_summary_result.cluster_area = cat(1,all_summary_result.cluster_area,summary_result.cluster_area(summary_result.cluster_index));
        elseif ismember(field_names{j},'cluster_info' )
            all_summary_result.cluster_info = cat(1,all_summary_result.cluster_info,summary_result.cluster_info(summary_result.cluster_index,:));
        else

            all_summary_result.(field_names{j}) = cat(1,all_summary_result.(field_names{j}),summary_result.(field_names{j}));
        end
    end
end


%%
n_rand = 1000;
min_length  = 0.25;
x_lim = [-1 2];
area_list = unique(all_summary_result.cluster_area);
onset_time = all_summary_result.onset_time(1:shortest_time);
beahvior_list = all_summary_result.staked_labels;
response_window = [0 min_length];
percentil_activation = 99;
for j=1:numel(area_list)
     figure('units','normalized','outerposition',[0 0 1 1]);
     sp = 1;
     fig_n =1;
    for bn = 1:numel(beahvior_list)-1
        if sp>6
            sp=1;
            sgtitle(area_list{j});
            saveas(gcf,[area_list{j}, '#', num2str(fig_n), '.jpg'])
            close(gcf)
            figure('units','normalized','outerposition',[0 0 1 1]);
            fig_n = fig_n+1;
        end
    index = ismember(all_summary_result.cluster_area, area_list{j});
    all_neurons_response = squeeze(all_summary_result.staked_responses(index,bn,:));
    session = all_summary_result.cluster_info.Session(index);
    [population_mean, population_pctl, excess_activation, excess_inhibition] = estiamte_activation(all_neurons_response,session,onset_time, response_window, n_rand, percentil_activation);


    response_index = mean(all_neurons_response(:,onset_time>response_window(1) & onset_time<response_window(2)),2);
    arr = session(:);  % Ensure column vector

    % Find where the value changes
    changeIdx = [1; find(diff(arr) ~= 0) + 1];
    endIdx = [changeIdx(2:end) - 1; length(arr)];

    values = arr(changeIdx);
    midpoints = (changeIdx + endIdx) / 2;

    % Compute mean index between previous group end and current group end
    prevBorders = [1; endIdx(1:end-1) + 1];
    meanIdx = (prevBorders + endIdx) / 2;

    % Create result table
    groupTable = table(values, endIdx+.5, meanIdx, ...
        'VariableNames', {'Value', 'MidpointIndex', 'MeanIndex'});

  
   colormap(jet)
   subplot(5,6, [1 7 13]+sp-1)
   re_arrange_order = nan(size(all_neurons_response,1),1);

   for sep = 1:size(groupTable,1)
       this_session = find(session==groupTable.Value(sep));
       [~, re_order] = sort(response_index(this_session));

       re_arrange_order(this_session) = this_session(re_order);
   end
   imagesc(onset_time , 1:size(all_neurons_response,1), all_neurons_response(re_arrange_order,:))
  
        hold on
    for sep = 1:size(groupTable,1)
        plot([onset_time(1) onset_time(end)], [1 1]*groupTable.MidpointIndex(sep), 'w', 'LineWidth',2)
    end
    plot([0 0],[1 size(all_neurons_response,1)], 'w', 'LineWidth',2)
    yticks(groupTable.MeanIndex)
    yticklabels(strsplit(num2str(groupTable.Value')))
   clim([-3 3])
   xlim(x_lim)
   axis xy
      xticklabels([])
   ylabel('Session Id')
    title(beahvior_list{bn})
   
   % 
   % stacked_activated = [];
   % for sep = 1:size(groupTable,1)
   %     this_session_activated =  all_neurons_response(session==groupTable.Value(sep) & response_index>=0,:);
   % 
   %     if size(this_session_activated,1)==1
   %         stacked_activated = [stacked_activated;this_session_activated];
   %     elseif size(this_session_activated,1)>1
   %         stacked_activated = [stacked_activated;mean(this_session_activated)];
   %     else
   %          stacked_activated = [stacked_activated;nan(1, numel(onset_time))];
   %     end
   % end
   % 
   % matrix2plot = stacked_activated;
   % this_sp = subplot(5,6, 19+sp-1);
   subplot(5,6, 19+sp-1);
    color ='r';
    plot(onset_time, population_mean(1,:)-excess_activation(1,:), color)
    hold on

   % [~, ~] = plot_ci(time,matrix2plot,this_sp, color, false);
   % is_significant = population_pctl(1,:)<0.01;
   % d = diff([0 is_significant 0]);       % pad with zeros to detect edges
   % starts = find(d == 1);   % rising edges (start of run)
   % ends = find(d == -1) - 1; % falling edges (end of run)  
   % y = y_lim(2);  % y-level for the lines
   % 
   % for i = 1:length(starts)
   %     if ends(i)-starts(i)>1
   %         plot(onset_time([starts(i), ends(i)]), [y y], 'r','LineWidth', 4);
   %     else
   %         plot(onset_time([starts(i), ends(i)]), [y y],'.r', 'MarkerSize',10 );
   %     end
   % end

   xlim(x_lim)
   ylim tight
   y_lim = ylim;
   plot([0 0], y_lim, 'k', 'LineWidth',2)
   plot(x_lim, [0  0], ':k')
   
   ylabel('excess activation')
   % 
   % stacked_inhibited = [];
   % for sep = 1:size(groupTable,1)
   %     this_session_activated =  all_neurons_response(session==groupTable.Value(sep) & response_index<0,:);
   % 
   %     if size(this_session_activated,1)==1
   %         stacked_inhibited = [stacked_inhibited;this_session_activated];
   %     elseif size(this_session_activated,1)>1
   %         stacked_inhibited = [stacked_inhibited;mean(this_session_activated)];
   %     else
   %          stacked_inhibited = [stacked_inhibited;nan(1, numel(onset_time))];
   %     end
   % end
   subplot(5,6, 25+sp-1);
   
   color ='b';
    plot(onset_time, population_mean(2,:)-excess_inhibition(2,:), color)
    hold on

   xlim(x_lim)
   ylim tight
   y_lim = ylim;
   plot([0 0], y_lim, 'k', 'LineWidth',2)
   plot(x_lim, [0  0], ':k')
   % matrix2plot = stacked_inhibited;
   % [mean_value, ci] = plot_ci(time,matrix2plot,this_sp, color, false);
   % xlim(x_lim)
   % ylim tight
   % y_lim = ylim;
   % 
   % is_significant = population_pctl(2,:)<0.01;
   % d = diff([0 is_significant 0]);       % pad with zeros to detect edges
   % starts = find(d == 1);   % rising edges (start of run)
   % ends = find(d == -1) - 1; % falling edges (end of run)  
   % y = y_lim(1);  % y-level for the lines
   % 
   % for i = 1:length(starts)
   %     if ends(i)-starts(i)>1
   %         plot(onset_time([starts(i), ends(i)]), [y y], 'b','LineWidth', 4);
   %     else
   %         plot(onset_time([starts(i), ends(i)]), [y y],'.b', 'MarkerSize',10 );
   %     end
   % end
   ylim tight
   ylabel('excess inhibition')
   xlabel('Time (s)')

    sp= sp+ 1;
    end

    sgtitle(area_list{j});
    saveas(gcf,[area_list{j}, '#', num2str(fig_n), '.jpg'])
    pause(.1)
    close(gcf)
end



