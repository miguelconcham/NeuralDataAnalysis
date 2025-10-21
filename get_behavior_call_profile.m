% function [] = get_behavior_call_profile(current_dir,behaviors2analyse,area2analyze,range_borders,psth_bin)



area2analyze      = 'PAG';
behaviors2analyse = {'Pounce_A','Pounce_Ai','Pounce_B','Pounce_Bi'};
behaviors2analyse = {'CC','CB','CD','Pounce_A','Pounce_B','Evasion', 'Escape', 'Pin', 'Box'};
current_dir = cd;
range_borders = [-2 2];
psth_bin = 0.001;
% range_borders     = [-1 1];
% psth_bin          = 0.01;


%% load synch
hmm_directory       = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\HMM data\HMM raw data';
synch_directory     = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Synch data';
area_limit_table    = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\Area_limits_GoodLooking.xlsx';
cd(current_dir)
animal_code         = strsplit(current_dir, '\');
animal_code         = animal_code{end};
animal_code_params  = strsplit(animal_code, ' ');
animal_batch        = animal_code_params{1};
date                = animal_code_params{2};
repeated_animal     = animal_code_params{3};

cd([synch_directory, '\', animal_code])
load('synch_model_video2NPX')
load('synch_model_audio2NPX')


%%  load behavior data

cd([hmm_directory, '\',animal_code])
disp('Loading Data')
Call_file   = dir('*.xlsx'); %load call data
Behavior_file = dir('*.txt') ;%load behavior data

CallStats                           = readtable(Call_file.name);
CallStats.Properties.VariableNames  = cellfun(@(x) strrep(x, '_', ''),CallStats.Properties.VariableNames, 'UniformOutput',false );
CallStats.BeginTimes                = predict(synch_model_audio2NPX, CallStats.BeginTimes);
CallStats.EndTimes                  = predict(synch_model_audio2NPX, CallStats.EndTimes);
Behavior                            = readtable(Behavior_file.name);
Behavior(:,2)                       = [];
Behavior.Properties.VariableNames   = {'Animal', 'Start', 'End', 'Length', 'Type'};
Partner_sessions  = Behavior{ismember(Behavior.Type,'Partners session'),{'Start','End'}};

ThihklingSession = [Behavior.Start( ismember(Behavior.Type, 'Tickling')) ...
                    Behavior.End( ismember(Behavior.Type, 'Tickling'))];

beh_bin = 0.01;
conv_length = 1;
Behavior.Type2 = Behavior.Type;
Behavior.Type2(ismember(Behavior.Type2, {'Pounce_A', 'Pounce_B'})) = {'Pounce'}; %% Merging behaviors to Type2
Behavior.Type2(ismember(Behavior.Type2, {'Pounce_Ai', 'Pounce_Bi'})) = {'PounceI'};
Behavior.Type2(ismember( Behavior.Type2,'')) = {'Other'};
Behavior(ismember(Behavior.Animal, 'Reversal'),:) = [];
animal_types = unique(Behavior.Animal);
animal_types(ismember(animal_types,'Session_structure'))=[];
Behavior.Start = predict(synch_model_video2NPX, Behavior.Start);
Behavior.End = predict(synch_model_video2NPX, Behavior.End);
CallStats.PrevCallDist = Inf(size(CallStats,1),1);
CallStats.PrevCallDist(2:end) = CallStats.BeginTimes(2:end)-CallStats.EndTimes(1:end-1);


%% load spikes

cd(current_dir)
area_limit = readtable(area_limit_table);
spike_times     = double(readNPY(['spike_times_', area2analyze, '.npy']))/30000;
spike_clusters  = readNPY(['spike_clusters_', area2analyze, '.npy']);
cluster_info    = readtable(['cluster_info_', area2analyze, '.tsv'] ,"FileType","text",'Delimiter', '\t');
cluster_info   = cluster_info(ismember(cluster_info.group,{'mua', 'good'}),:);
channel_map_loaded = false;
if exist('channel_positions.npy ','file')>0
    channel_map_loaded =true;
    channel_positions = readNPY('channel_positions.npy ');
    channel_map = readNPY('channel_map.npy');
    cluster_channels = cluster_info.ch;
    hard_coded_x_coords = [8 40;258 290; 508 540; 758 790];
   
end



cluster_area = cell(size(cluster_info,1),1);
if strcmp(animal_batch(3), 'D')
    animal_type = 'Dual';
else
    animal_type = 'Single';
end

this_animal_id = ['Batch', animal_batch(2), animal_type, animal_batch(4)];

for j=1:numel(cluster_area)
    if channel_map_loaded
        this_ch_pos =  channel_positions(ismember(channel_map,cluster_channels(j)));

        probe_n = find(any(ismember(hard_coded_x_coords,this_ch_pos),2));
        cluster_area(j) = area_limit.area(area_limit.depth_start<=cluster_info.depth(j) & area_limit.depth_end>=cluster_info.depth(j) & ismember(area_limit.AnimalName, this_animal_id) & area_limit.ProbeNum==probe_n);
    else
    cluster_area(j) = area_limit.area(area_limit.depth_start<=cluster_info.depth(j) & area_limit.depth_end>=cluster_info.depth(j) & ismember(area_limit.AnimalName, this_animal_id));
    end
end

%% definin call per type

Partner_sessions  = Behavior{ismember(Behavior.Type,'Partners session'),{'Start','End'}};

if isempty(ThihklingSession)
    ThihklingSession = [Inf Inf];
end
thickling_call_index = find(CallStats.BeginTimes>=ThihklingSession(1) & CallStats.EndTimes<=ThihklingSession(2));




%% plot calls these behaviors



behaviors2analyse = {'CC', 'CB', 'CD', 'Pounce_A', 'Pounce_B', 'Pin', 'Boxing'};

% beh_indexes = find(ismember(Behavior.Type, behaviors2analyse) & ismember(Behavior.Animal, repeated_animal));
beh_indexes = find(ismember(Behavior.Type, behaviors2analyse) );

[ordered_beh_length,beh_length_order] = sort(Behavior.End(beh_indexes) - Behavior.Start(beh_indexes));
beh_indexes = beh_indexes(beh_length_order);


psth_edges = -5:0.01:10;
psth_calls_onset = zeros(numel(beh_indexes),numel(psth_edges));

calls_within = [];
behavior_type = {};
for j=1:numel(beh_indexes)
    beh_start = Behavior.Start(beh_indexes(j));
    beh_end   = Behavior.End(beh_indexes(j));
    this_beh_length = beh_end-beh_start;
    this_behavior_type = Behavior.Type{beh_indexes(j)};

    these_calls = find(CallStats.EndTimes>=beh_start+range_borders(1) & CallStats.BeginTimes<=beh_start+this_beh_length+range_borders(2));

    for cn = 1:numel(these_calls)
        call_beg  = CallStats.BeginTimes(these_calls(cn));
        call_end  = CallStats.EndTimes(these_calls(cn));

        if (call_end>=beh_start && call_end<=beh_end) || ...
                (call_beg>=beh_start && call_beg<=beh_end)
            calls_within = [calls_within;[these_calls(cn), this_beh_length]];
            behavior_type = [behavior_type,this_behavior_type];
        end

           
        psth_calls_onset(j,psth_edges>=call_beg-beh_start & psth_edges<=call_end-beh_start) = 1;       


    end
end

%% estiamte cost between call during thikiling and during behavior 
% properties2compare =   {'CallLengths'};
properties2compare =   {'CallLengths', 'PrincipalFrequencykHz', 'SlopekHzs'};

call_properties     = CallStats{ :,properties2compare};
zscored_properties  = call_properties;
for j=1:size(zscored_properties,2)
    zscored_properties(:,j) = zscore(zscored_properties(:,j));
end

[~, pca_cata] = pca(zscored_properties);
calls_within_index      = calls_within(:,1);
thickling_call_properties   = zscored_properties(thickling_call_index, :);
behavior_call_properties    = zscored_properties(calls_within_index,:);

Cost = (thickling_call_properties(:,1)-behavior_call_properties(:,1)').^2;
for j=2:numel(properties2compare)
    Cost =Cost +  (thickling_call_properties(:,j)-behavior_call_properties(:,j)').^2;
end
Cost = sqrt(Cost);
%% find best match between behavior and thikling
costUnmatched =10;


figure
p = 0;
n_perm = 1000;
while p<0.5




    M = matchpairs(Cost,costUnmatched); %first colum is withplay second column witohuhtplay

    for j=1:numel(properties2compare)
        subplot(1, numel(properties2compare)+1, j)
        plot( call_properties(thickling_call_index(M(:,1)),j), call_properties( calls_within_index(M(:,2)),j), '.')
    end
    subplot(1, numel(properties2compare)+1, j+1)
    plot( pca_cata( thickling_call_index(M(:,1)),1), pca_cata( calls_within_index(M(:,2)),1), '.')
    pause(.1)
    % [p, real_E, perm_E] = test_difference( call_properties(thickling_call_index(M(:,1)),j), call_properties( calls_within_index(M(:,2)),j), n_perm)
    [h,p]= kstest2(pca_cata( thickling_call_index(M(:,1)),1), pca_cata( calls_within_index(M(:,2)),1));
    costUnmatched = costUnmatched/1.05;
end





%% estiamte call response sall neurons, all calls (duting behavior an thikling)

all_call_indexes = [thickling_call_index(M(:,1));calls_within_index(M(:,2))];
longest_call = max(CallStats.EndTimes(all_call_indexes)- CallStats.BeginTimes(all_call_indexes));
psth_edges_onset    = range_borders(1):psth_bin:ceil(range_borders(2)/psth_bin + longest_call/psth_bin)*psth_bin;
psth_edges_offset   = -ceil(range_borders(1)/psth_bin + longest_call/psth_bin)*psth_bin:psth_bin:range_borders(2);
onset_time = .5*(psth_edges_onset(1:end-1) + psth_edges_onset(2:end));

psth_tensor_onset  = nan(size(cluster_info,1),numel(all_call_indexes), numel(psth_edges_onset)-1);
psth_tensor_offset = nan(size(cluster_info,1),numel(all_call_indexes), numel(psth_edges_offset)-1);
mean_rate_during_call = nan(size(cluster_info,1),numel(all_call_indexes),2);
normalized_rate_during_call = nan(size(cluster_info,1),numel(all_call_indexes));

for cn=1:size(cluster_info,1)
    cluster_id =cluster_info.cluster_id(cn);
    this_neuron_spikes = spike_times(spike_clusters==cluster_id);

    spikes_this_session =this_neuron_spikes(any(this_neuron_spikes'>=Partner_sessions(:,1) & this_neuron_spikes'<=Partner_sessions(:,2),1)) ;
    
    partner_session_edges = floor(min(Partner_sessions(:,1))):ceil(max(Partner_sessions(:,2)));


    rate_distribution = histcounts(spikes_this_session,partner_session_edges);
    rate_distribution= rate_distribution(any(partner_session_edges(1:end-1)>=floor(Partner_sessions(:,1))  & partner_session_edges(2:end)<=ceil(Partner_sessions(:,2)),1)) ;

    rate_distribution = sort(rate_distribution);
    [rate_count,rate_val] = groupcounts(rate_distribution');
    rate_count = cumsum(rate_count)/sum(rate_count);
    rate_count =1 - rate_count/sum(rate_count);
    rate_count= interp1(rate_val,rate_count,min(rate_val):max(rate_val));
    rate_val = min(rate_val):max(rate_val);
    for bn = 1:numel(all_call_indexes)

        call_start   = CallStats.BeginTimes(all_call_indexes(bn));
        call_end     = CallStats.EndTimes(all_call_indexes(bn));
        mean_rate_during_call(cn,bn,1) = sum(this_neuron_spikes>=call_start & this_neuron_spikes<=call_end)/(call_end-call_start);

         mean_rate_during_call(cn,bn,2) = sum(this_neuron_spikes>=call_start-5 & this_neuron_spikes<=call_start)/5;
        if  mean_rate_during_call(cn,bn,1)>max(rate_val)
            normalized_rate_during_call(cn,bn)  = 1;
        elseif mean_rate_during_call(cn,bn,1)<min(rate_val)
            normalized_rate_during_call(cn,bn) = 0;
        else
            normalized_rate_during_call(cn,bn) = rate_count(rate_val == floor(mean_rate_during_call(cn,bn,1)) );
        end
        this_call_spikes = this_neuron_spikes(this_neuron_spikes>=call_start+range_borders(1) & this_neuron_spikes<=call_end+range_borders(2));
        psth_tensor_onset(cn,bn,:) = histcounts(this_call_spikes-call_start,psth_edges_onset );
        psth_tensor_onset(cn,bn,psth_edges_onset(2:end)>call_end-call_start+range_borders(2))=NaN;



          psth_tensor_offset(cn,bn,:) = histcounts(this_call_spikes-call_end,psth_edges_offset );
        psth_tensor_offset(cn,bn,psth_edges_offset(1:end-1)<call_start-call_end-range_borders(1))=NaN;


    end
   

end

call_length = CallStats.EndTimes(all_call_indexes) - CallStats.BeginTimes(all_call_indexes);
PrevCallDist = CallStats.PrevCallDist(all_call_indexes);
% end

is_thicklish = false(2*size(M,1),1);
is_thicklish(1:size(M,1)) = true;

%% plot call responses if needed

min_length = .04;
min_prev_dist = 0.1
time2plot_index = onset_time >=-min_prev_dist & onset_time <2*min_length;
trim_range = 0;
figure('units','normalized','outerposition',[0 0 1 1]);
sp = 1;
n_col = 10;
mov_mean_wind = 100;
nr = 0;
fig_n = 1;
for cn = 1:106
    if sp>n_col
        sp = 1;
        pause(.1)
        saveas(gcf,['sumary figure#', num2str(fig_n), '.jpg'])
        fig_n = fig_n+1;
        close(gcf)
        figure('units','normalized','outerposition',[0 0 1 1]);
    end
    this_psth = squeeze(psth_tensor_onset(cn,:,:));
    onset_time = .5*(psth_edges_onset(1:end-1) + psth_edges_onset(2:end));
    



    thicklish_indexes =sub_calls;
    thicklish_indexes(sorted_distances(1:nr))=[];
    beh_indexes = thicklish_indexes + sum(is_thicklish);

    colormap(1-gray)
    subplot(3,n_col,sp)
    % index = is_thicklish & call_length>min_length & PrevCallDist>min_prev_dist;
    index = thicklish_indexes;
    this_length = call_length(index);
    this_matrix_th = this_psth(index ,:);
    [this_sorted_lengths,this_length_order] = sort(this_length);
    are_there_spikes = any(this_matrix_th(:,time2plot_index)>0,2);
    are_there_spikes = are_there_spikes(this_length_order);
    are_there_spikes = true(size(this_sorted_lengths));
    matrix2imagesc = this_matrix_th(this_length_order(are_there_spikes),:);
    for j=1:size(matrix2imagesc,1)
        matrix2imagesc(j,:) = movmean(matrix2imagesc(j,:), mov_mean_wind/25);
    end
    imagesc(onset_time,1:sum(are_there_spikes),matrix2imagesc)
    hold on
    plot(this_sorted_lengths(are_there_spikes),1:sum(are_there_spikes), 'b' )
    plot(this_sorted_lengths(are_there_spikes)*0,1:sum(are_there_spikes), 'b' )
    axis xy
    yticks([])
    xlim([-min_prev_dist 2*min_length])
    title([cluster_area{cn}, ' ', num2str(cn)])

    subplot(3,n_col,n_col+sp)
    % index = ~is_thicklish & call_length>min_length & PrevCallDist>min_prev_dist;
    index =beh_indexes;
    this_length = call_length(index);
    [this_sorted_lengths,this_length_order] = sort(this_length);

    this_matrix_beh = this_psth(index ,:);
    [this_sorted_lengths,this_length_order] = sort(this_length);
    % are_there_spikes = any(this_matrix_beh(:,time2plot_index)>0,2);
    % are_there_spikes = are_there_spikes(this_length_order);
    are_there_spikes = true(size(this_sorted_lengths));

     matrix2imagesc = this_matrix_beh(this_length_order(are_there_spikes),:);
    for j=1:size(matrix2imagesc,1)
        matrix2imagesc(j,:) = movmean(matrix2imagesc(j,:), mov_mean_wind/25);
    end
    imagesc(onset_time,1:sum(are_there_spikes),matrix2imagesc)
    hold on
    plot(this_sorted_lengths(are_there_spikes),1:sum(are_there_spikes), 'r' )
    plot(this_sorted_lengths(are_there_spikes)*0,1:sum(are_there_spikes), 'r' )
    axis xy
    xlim([-min_prev_dist 2*min_length])
    yticks([])


    subplot(3,n_col,2*n_col +sp)
    promedio_th = nan(1,size(this_matrix_th,2));
    for t=1:size(this_matrix_th,2)
        no_nan = ~isnan(this_matrix_th(:,t));
        promedio_th(t) =  trimmean(this_matrix_th(no_nan, t), trim_range);
    end
    promedio_th = movmean(promedio_th, mov_mean_wind)/psth_bin;
    plot(onset_time,promedio_th, 'b')
    hold on
    promedio_bh = nan(1,size(this_matrix_beh,2));
    for t=1:size(this_matrix_beh,2)
        no_nan = ~isnan(this_matrix_beh(:,t));
        promedio_bh(t) =  trimmean(this_matrix_beh(no_nan, t), trim_range);
    end
      promedio_bh = movmean( promedio_bh, mov_mean_wind)/psth_bin;
    plot(onset_time, promedio_bh, 'r')
    xlim([-min_prev_dist 2*min_length])
    y_lim = ylim;
    plot([0 0], y_lim, 'k', 'HandleVisibility','off')
    % yticklabels([])
    pause(.1)
    % legend({'Thikling', behaviors2analyse{1}})

    sp = sp+1;
end



%% select calls without prev calls befroe min_prev_dist, longer than min_length, and estiamte call response for two conditions (behavior and thilling)


sigma       = 40;                          % Standard deviation (in samples)
win_size    = 5 * sigma;              % Total window size (ensure it's odd)
g           = gausswin(win_size);            % Generate 1D Gaussian window
g           = g / sum(g);     

convoluted_trail_data = psth_tensor_onset;

for cn=1:size(convoluted_trail_data,1)
    for call =1:size(convoluted_trail_data,2)        
        convoluted_trail_data(cn,j,:) =  conv(squeeze(convoluted_trail_data(cn,j,:)), g, 'same') ;
    end
end


min_length      = .02;
min_prev_dist   = 0.05;


chose_time_to_compare   = onset_time>-min_prev_dist & onset_time<min_length;
sub_calls               = find(call_length(is_thicklish)>min_length &  call_length(~is_thicklish)>min_length & ...
                        PrevCallDist(is_thicklish)>min_prev_dist &  PrevCallDist(~is_thicklish)>min_prev_dist);

thicklish_indexes       = sub_calls;
beh_indexes             = thicklish_indexes + sum(is_thicklish);
call_length             = CallStats.EndTimes(all_call_indexes) - CallStats.BeginTimes(all_call_indexes);

thicklish_matrix        = convoluted_trail_data(:, thicklish_indexes,chose_time_to_compare);
behavior_matrix         = convoluted_trail_data(:, beh_indexes,chose_time_to_compare);
cosine_projections      = nan(size(thicklish_matrix,[1 2]));
rate_difference         = nan(size(thicklish_matrix,[1 2]));

for neuron_n = 1:size(thicklish_matrix,1)
    for call_n = 1:size(thicklish_matrix,2)
        A = squeeze(thicklish_matrix(neuron_n, call_n,:));
        B = squeeze(behavior_matrix(neuron_n, call_n,:));
        if sum(A)==0 && sum(B)==0
            rate_difference(neuron_n,call_n) = 0;
            cosine_projections(neuron_n,call_n) = 1;
        elseif sum(A)==0 || sum(B)==0
            cosine_projections(neuron_n,call_n) = 0;
            rate_difference(neuron_n,call_n) = sum(abs(A-B))/(sum((A+B)/2));
        else
            cosine_projections(neuron_n,call_n) = sum(A.*B, 'omitmissing')/(norm(A(~isnan(A)))*norm(B(~isnan(B))));
            rate_difference(neuron_n,call_n)    = sum(abs(A-B))/(sum((A+B)/2));
        end
    end
end

% mean_distance = mean(cosine_projections);
mean_distance = mean(rate_difference);
[distance_value_sorted, sorted_distances] = sort(mean_distance, 'descend');

%% evaluate/estiamte  call response diferences beteen behavior and thikling (Can take several minutes)

figure
n_iterations = round(numel(sub_calls)*.75);
% % progress_distance = nan(size(convoluted_trail_data,1),n_iterations);
% rate_progress_distance = nan(2,size(convoluted_trail_data,1),n_iterations);
trim_range = 0;
for nr = 287:n_iterations
    disp(nr)
    thicklish_indexes =sub_calls;
    thicklish_indexes(sorted_distances(1:nr))=[];
    beh_indexes = thicklish_indexes + sum(is_thicklish);

    for cn=1:size(convoluted_trail_data,1)
        if nr==1 || max(rate_progress_distance(:,cn,nr-1))>0
            this_psth = squeeze(psth_tensor_onset(cn,:,:));
            this_matrix_th = this_psth(thicklish_indexes ,:);
            this_matrix_beh = this_psth(beh_indexes ,:);
            promedio_th = nan(1,size(this_matrix_th,2));
            for t=1:size(this_matrix_th,2)
                no_nan = ~isnan(this_matrix_th(:,t));
                promedio_th(t) =  trimmean(this_matrix_th(no_nan, t), trim_range);
            end
            promedio_th = movmean(promedio_th, 10)/psth_bin;

            promedio_bh = nan(1,size(this_matrix_beh,2));
            for t=1:size(this_matrix_beh,2)
                no_nan = ~isnan(this_matrix_beh(:,t));
                promedio_bh(t) =  trimmean(this_matrix_beh(no_nan, t), trim_range);
            end
            promedio_bh = movmean( promedio_bh, 10)/psth_bin;
            progress_distance(cn,nr) = sum(abs(promedio_th(chose_time_to_compare)-promedio_bh(chose_time_to_compare)), 'omitmissing');
            rate_progress_distance(1,cn,nr)  = sum(promedio_bh(chose_time_to_compare));
            rate_progress_distance(2,cn,nr)  = sum(promedio_th(chose_time_to_compare));
        else
            disp(['Neuron ', num2str(cn), ' reached 0 rate'])
        end
    end
    hold off
    normalizing_diag = diag(1./progress_distance(:,1));
        normalizing_diag(isinf(normalizing_diag)) = 1;
        non_nan_progress_distance = progress_distance(:, 1:nr);
    non_nan_progress_distance(isnan(non_nan_progress_distance)) = 0;

    non_nan_progress_distance = non_nan_progress_distance(:, 1:nr);
     plot(100*(1:nr)/numel(sub_calls), (normalizing_diag*non_nan_progress_distance(:, 1:nr))', 'k')
    hold on
    plot(100*(1:nr)/numel(sub_calls),mean(normalizing_diag*non_nan_progress_distance),  'r', 'LineWidth',2)
    pause(.1)
end

%% plot difference evolution over time
 
centered_Data = (normalizing_diag*non_nan_progress_distance(:, 1:nr)) -1;
 figure
  hold on
 for cn=1:size(centered_Data,1)
      first_zero = min(find( any(isnan(squeeze(rate_progress_distance(:,cn,:))),1)));
      centered_Data(cn,first_zero:end) = NaN;
  plot(100*(1:nr)/numel(sub_calls), centered_Data(cn,:), 'k')
 end
   
    plot(100*(1:nr)/numel(sub_calls),mean(centered_Data, 'omitmissing'),  'r', 'LineWidth',2)


    for cn=1:size(centered_Data,1)
        first_zero = min(find( any(isnan(squeeze(rate_progress_distance(:,cn,:))),1)));
         if ~isempty(first_zero)
        plot(100*(first_zero-1)/numel(sub_calls), centered_Data(cn,first_zero-1), 'b.', 'MarkerSize',20)
         end

    end



%% find variation distribution: 1 create pair of similar calls within thikling


thickling_call_properties   = zscored_properties(thickling_call_index, :);

Cost_within = (thickling_call_properties(:,1)-thickling_call_properties(:,1)').^2;
for j=2:numel(properties2compare)
    Cost_within =Cost_within +  (thickling_call_properties(:,j)-thickling_call_properties(:,j)').^2;
end
Cost_within = sqrt(Cost_within);



D = Cost_within;
n = size(Cost_within, 1);
  

    % Set diagonal to Inf to avoid self-pairs
    D(logical(eye(n))) = Inf;

    % Use matchpairs to find minimal-cost pairing
    pairs = matchpairs(D, 1e10, 'min');

    % Ensure pairs are unordered and unique
    pairs = sort(pairs, 2);           % Sort each pair (e.g. [3 1] -> [1 3])
    pairs = unique(pairs, 'rows');  

    is_duplicated = zeros(size(pairs,1),1);

    for j=1:size(pairs,1)
        for k=j+1:size(pairs,1)

            if pairs(j,:) == pairs(k,[2 1])
                is_duplicated(j) = k;
            end
        end
    end
               



% figure
% p = 0;
% n_perm = 1000;
% while p<0.5




%% plot matched calls

figure
% M = matchpairs(Cost,costUnmatched); %first colum is withplay second column witohuhtplay

for j=1:numel(properties2compare)
    subplot(1, numel(properties2compare)+1, j)
    plot( call_properties(thickling_call_index(pairs(:,1)),j), call_properties( thickling_call_index(pairs(:,2)),j), '.')
end
subplot(1, numel(properties2compare)+1, j+1)
plot( pca_cata( thickling_call_index(pairs(:,2)),1), pca_cata( thickling_call_index(pairs(:,2)),1), '.')
pause(.1)
% [p, real_E, perm_E] = test_difference( call_properties(thickling_call_index(M(:,1)),j), call_properties( calls_within_index(M(:,2)),j), n_perm)
[h,p]= kstest2(pca_cata( thickling_call_index(M(:,1)),1), pca_cata( calls_within_index(M(:,2)),1));
costUnmatched = costUnmatched/1.05;
% end
    
 %% find avraition distribution: 2 estiamte psth onset for all selected thickilng calls

all_call_indexes4distr      = thickling_call_index(pairs(:));
longest_call                = max(CallStats.EndTimes(all_call_indexes4distr)- CallStats.BeginTimes(all_call_indexes4distr));
psth_edges_onset4distr            = range_borders(1):psth_bin:ceil(range_borders(2)/psth_bin + longest_call/psth_bin)*psth_bin;
onset_time4distr            = .5*(psth_edges_onset4distr(1:end-1) + psth_edges_onset4distr(2:end));

psth_tensor_onset4distr     = nan(size(cluster_info,1),numel(all_call_indexes4distr), numel(psth_edges_onset4distr)-1);   



for cn=1:size(cluster_info,1)
    cluster_id =cluster_info.cluster_id(cn);
    this_neuron_spikes = spike_times(spike_clusters==cluster_id);

    for call_n = 1:numel(all_call_indexes4distr)

        call_start   = CallStats.BeginTimes(all_call_indexes4distr(call_n));
        call_end     = CallStats.EndTimes(all_call_indexes4distr(call_n));

        this_call_spikes = this_neuron_spikes(this_neuron_spikes>=call_start+range_borders(1) & this_neuron_spikes<=call_end+range_borders(2));
        psth_tensor_onset4distr(cn,call_n,:) = histcounts(this_call_spikes-call_start,psth_edges_onset );


    end
   

end


%%
n_randomizations = 1000;

pctl2check = [1 .9 .8 .7 .6 .5 .4 .3 .25];
points2cehck = round(numel(sub_calls)*pctl2check);
points2cehck = sort(points2cehck, 'descend');
% progress_distance_distr = nan(numel(points2cehck), size(psth_tensor_onset4distr,1),n_randomizations);
% all_combos = cell(numel(points2cehck),1);
progress_distance_distr(1:6,:,:) = progress_distance_distr_aux;
for point_n = 7:numel(points2cehck)
    current_point2check =  points2cehck(point_n);


    combos = zeros(n_randomizations, points2cehck(point_n));
    for i = 1:n_randomizations
        combos(i, :) = sort(randperm(size(pairs,1),  points2cehck(point_n)));  % Choose k unique elements
    end
    all_combos{point_n} = combos;


    % progress_distance_distr = nan(size(psth_tensor_onset4distr,1),n_randomizations);
    for cn = 1:size(psth_tensor_onset4distr,1)
        disp([point_n cn])
        this_psth = squeeze(psth_tensor_onset4distr(cn,:,:));
        for i= 1:n_randomizations
            this_matrix_th1 = this_psth(pairs(combos(i,:),1) ,:);
            this_matrix_th2 = this_psth(pairs(combos(i,:),2) ,:);
            promedio_th1 = nan(1,size(this_matrix_th1,2)d);
            for t=1:size(this_matrix_th1,2)
                no_nan = ~isnan(this_matrix_th1(:,t));
                promedio_th1(t) =  trimmean(this_matrix_th1(no_nan, t), trim_range);
            end
            promedio_th1 = movmean(promedio_th1, 10)/psth_bin;

            promedio_th2 = nan(1,size(this_matrix_th2,2));
            for t=1:size(this_matrix_th2,2)
                no_nan = ~isnan(this_matrix_th2(:,t));
                promedio_th2(t) =  trimmean(this_matrix_th2(no_nan, t), trim_range);
            end
            promedio_th2 = movmean( promedio_th2, 10)/psth_bin;
            progress_distance_distr(point_n,cn,i) = sum(abs(promedio_th1(chose_time_to_compare)-promedio_th2(chose_time_to_compare)), 'omitmissing');
        end

        pctiles5_95= prctile(squeeze(progress_distance_distr(point_n,cn,:))', [5 95]);

        figure
        plot(100*(1:n_iterations)/numel(sub_calls),squeeze(progress_distance(cn,:)), 'k')
        hold on
        current_estiamte = 100*(numel(sub_calls)-current_point2check + [-10 10])/numel(sub_calls);
        fill(current_estiamte([1 2 2 1]), pctiles5_95([1 1 2 2]), 'r', 'FaceAlpha', .25, 'EdgeColor','none')
        title(cn)
        pause(.1)
    end
    save('voc_th_beh_data','progress_distance_distr','progress_distance','rate_progress_distance','n_iterations','min_prev_dist','min_length','onset_time','sub_calls','all_call_indexes','rate_difference','cosine_projections','points2cehck', 'all_combos')

end

%%
 figure
 sp =1;
 for cn =60:90;

     if sp>16
         figure
         sp=1;
     end
     subplot(4,4,sp)
   
      
     starting_Rate = progress_distance(cn,1);
     plot(100*(1:n_iterations)/numel(sub_calls),(squeeze(progress_distance(cn,:))/starting_Rate) -1, '-k')
     hold on
     for point_n=1:numel(points2cehck)
         current_point2check =  points2cehck(point_n);

         pctiles5_95= prctile(squeeze(progress_distance_distr(point_n,cn,:))', [10 90])/starting_Rate - 1;
         current_estiamte = 100*(numel(sub_calls)-current_point2check + [-10 10])/numel(sub_calls);

         fill(current_estiamte([1 2 2 1]), pctiles5_95([1 1 2 2]), 'r', 'FaceAlpha', .25, 'EdgeColor','none')
         pctiles5_95= prctile(squeeze(progress_distance_distr(point_n,cn,:))', [5 95])/starting_Rate - 1;
         current_estiamte = 100*(numel(sub_calls)-current_point2check + [-10 10])/numel(sub_calls);

         fill(current_estiamte([1 2 2 1]), pctiles5_95([1 1 2 2]), 'r', 'FaceAlpha', .25, 'EdgeColor','none')
         pctiles5_95= prctile(squeeze(progress_distance_distr(point_n,cn,:))', [1 99])/starting_Rate - 1;
         current_estiamte = 100*(numel(sub_calls)-current_point2check + [-10 10])/numel(sub_calls);

         fill(current_estiamte([1 2 2 1]), pctiles5_95([1 1 2 2]), 'r', 'FaceAlpha', .25, 'EdgeColor','none')
       
     end
       xlim([-1 75])
         y_lim = ylim;   

         plot(repmat(100 - 100*pctl2check,2,1), y_lim, '-r')
            yyaxis right

            plot(100*(1:n_iterations)/numel(sub_calls),squeeze(rate_progress_distance(1,cn,:)), '-b')
         hold on
         plot(100*(1:n_iterations)/numel(sub_calls),squeeze(rate_progress_distance(2,cn,:)), '-m')
     title([cluster_area{cn}, ' ', num2str(cn)])
     pause(.1)
     sp = sp+1;
 end

    %% ideitify differnece profiles and find thee neurons

 
 
 iterations2classify = round(numel(sub_calls)/2);
last5 = mean(centered_Data(:,end-4:end),2);
[~, sortedbyend] = sort(last5);
[coeff,score,latent,tsquared,explained,mu] = pca(centered_Data);



%%
figure
colormap(jet)
imagesc(100*first_zero/numel(sub_calls), 1:size(centered_Data,1),centered_Data(sortedbyend,:))
hold on
axis xy
clim([-1 3])

sorted_Rate_progress = rate_progress_distance(:,sortedbyend,:);
for cn=1:size(centered_Data,1)
    first_zero = min(find( any(isnan(squeeze(sorted_Rate_progress(:,cn,:))),1)));
    if ~isempty(first_zero)
        plot(100*first_zero/numel(sub_calls),cn , 'w.', 'MarkerSize',20)
    end

end
%% how about area distribution

increasing_range = mean(centered_Data(:,iterations2classify:end),2, 'omitmissing');
[a_non_vocal,b_non_vocal_Area]=groupcounts(cluster_area(increasing_range>0 & ~any(isnan(centered_Data),2)));

[a_vocal,b_vocal_area]=groupcounts(cluster_area(increasing_range<0 & ~any(isnan(centered_Data),2)));

stacked_bar = [a_non_vocal,a_vocal];

figure
bar(diag(1./sum(stacked_bar,2))*stacked_bar, 'stacked')
xticks(1:numel(a_vocal))
xticklabels(b_vocal_area)
legend({'Non_vocal','Vocal'})
%%
% sorted_ny_mean_end = mean()

figure
plot(progress_distance(2,:))
plot(progress_distance(18,:))

%% plot evolution of respones for a single neuronr
figure('units','normalized','outerposition',[0 0 1 1]);
n_col = 10;
trim_range = 0;
cn = 47;
sp = 1;
for nr = round(linspace(1, n_iterations,n_col))

  
    thicklish_indexes =sub_calls;
    thicklish_indexes(sorted_distances(1:nr))=[];
    beh_indexes = thicklish_indexes + sum(is_thicklish);

    this_psth = squeeze(psth_tensor_onset(cn,:,:));
    onset_time = .5*(psth_edges_onset(1:end-1) + psth_edges_onset(2:end));

    colormap(1-gray)
    subplot(3,n_col,sp)
    % index = is_thicklish & call_length>min_length & PrevCallDist>min_prev_dist;
    index = thicklish_indexes;
    this_length = call_length(index);
    this_matrix_th = this_psth(index ,:);
    [this_sorted_lengths,this_length_order] = sort(this_length);
    imagesc(onset_time,1:numel(index),this_matrix_th(this_length_order,:))
    hold on
    plot(this_sorted_lengths,1:numel(index), 'b' )
    plot(this_sorted_lengths*0,1:numel(index), 'b' )
    axis xy
    yticks([])
    xlim([-min_prev_dist min_length])
    title(numel(index))

    subplot(3,n_col,n_col+sp)
    % index = ~is_thicklish & call_length>min_length & PrevCallDist>min_prev_dist;
    index = beh_indexes;
    this_length = call_length(index);
    [this_sorted_lengths,this_length_order] = sort(this_length);

    this_matrix_beh = this_psth(index ,:);
    imagesc(onset_time,1:numel(index),this_matrix_beh(this_length_order,:))
    hold on
    plot(this_sorted_lengths,1:numel(index), 'r' )
    plot(this_sorted_lengths*0,1:numel(index), 'r' )
    axis xy
    xlim([-min_prev_dist min_length])
    yticks([])


    subplot(3,n_col,2*n_col +sp)
    promedio_th = nan(1,size(this_matrix_th,2));
    for t=1:size(this_matrix_th,2)
        no_nan = ~isnan(this_matrix_th(:,t));
        promedio_th(t) =  trimmean(this_matrix_th(no_nan, t), trim_range);
    end
    promedio_th = movmean(promedio_th, 10)/psth_bin;
    plot(onset_time,promedio_th, 'b')
    hold on
    promedio_bh = nan(1,size(this_matrix_beh,2));
    for t=1:size(this_matrix_beh,2)
        no_nan = ~isnan(this_matrix_beh(:,t));
        promedio_bh(t) =  trimmean(this_matrix_beh(no_nan, t), trim_range);
    end
      promedio_bh = movmean( promedio_bh, 10)/psth_bin;
    plot(onset_time, promedio_bh, 'r')
    xlim([-min_prev_dist min_length])
    y_lim = ylim;
    plot([0 0], y_lim, 'k', 'HandleVisibility','off')
    re_Estiamte = 2*sum(abs(promedio_th(chose_time_to_compare)-promedio_bh(chose_time_to_compare)), 'omitmissing')/(sum(promedio_th(chose_time_to_compare)) + sum(promedio_bh(chose_time_to_compare)));
    title(num2str([re_Estiamte progress_distance(cn, nr)]))
    % yticklabels([])
    pause(.1)
    % legend({'Thikling', behaviors2analyse{1}})

    sp = sp+1;

end


%%

figure('units','normalized','outerposition',[0 0 1 1]);
n_col = 10;
trim_range = 0;
cn = 51;
sp = 1;

  this_psth = squeeze(psth_tensor_onset(cn,:,:));
    onset_time = .5*(psth_edges_onset(1:end-1) + psth_edges_onset(2:end));
for nr = round(linspace(1, n_iterations,n_col))

    thicklish_indexes =sub_calls;
    thicklish_indexes(sorted_distances(1:nr))=[];
    beh_indexes = thicklish_indexes + sum(is_thicklish);

    colormap(1-gray)
    subplot(3,n_col,sp)
    % index = is_thicklish & call_length>min_length & PrevCallDist>min_prev_dist;
    index = thicklish_indexes;
    this_length = call_length(index);
    this_matrix_th = this_psth(index ,:);
    [this_sorted_lengths,this_length_order] = sort(this_length);
    are_there_spikes = any(this_matrix_th(:,time2plot_index)>0,2);
    are_there_spikes = are_there_spikes(this_length_order);
    are_there_spikes = true(size(this_sorted_lengths));
    matrix2imagesc = this_matrix_th(this_length_order(are_there_spikes),:);
    for j=1:size(matrix2imagesc,1)
        matrix2imagesc(j,:) = movmean(matrix2imagesc(j,:), mov_mean_wind/25);
    end
    imagesc(onset_time,1:sum(are_there_spikes),matrix2imagesc)
    hold on
    plot(this_sorted_lengths(are_there_spikes),1:sum(are_there_spikes), 'b' )
    plot(this_sorted_lengths(are_there_spikes)*0,1:sum(are_there_spikes), 'b' )
    axis xy
    yticks([])
    xlim([-min_prev_dist 2*min_length])
    ylabel('Thickling')
    title([cluster_area{cn}, ' ', num2str(cn)])

    subplot(3,n_col,n_col+sp)
    % index = ~is_thicklish & call_length>min_length & PrevCallDist>min_prev_dist;
    index =beh_indexes;
    this_length = call_length(index);
    [this_sorted_lengths,this_length_order] = sort(this_length);

    this_matrix_beh = this_psth(index ,:);
    [this_sorted_lengths,this_length_order] = sort(this_length);
    % are_there_spikes = any(this_matrix_beh(:,time2plot_index)>0,2);
    % are_there_spikes = are_there_spikes(this_length_order);
    are_there_spikes = true(size(this_sorted_lengths));

     matrix2imagesc = this_matrix_beh(this_length_order(are_there_spikes),:);
    for j=1:size(matrix2imagesc,1)
        matrix2imagesc(j,:) = movmean(matrix2imagesc(j,:), mov_mean_wind/25);
    end
    imagesc(onset_time,1:sum(are_there_spikes),matrix2imagesc)
    hold on
    plot(this_sorted_lengths(are_there_spikes),1:sum(are_there_spikes), 'r' )
    plot(this_sorted_lengths(are_there_spikes)*0,1:sum(are_there_spikes), 'r' )
    axis xy
    xlim([-min_prev_dist 2*min_length])
    yticks([])
    ylabel('Play Behaviors')



    subplot(3,n_col,2*n_col +sp)
    promedio_th = nan(1,size(this_matrix_th,2));
    for t=1:size(this_matrix_th,2)
        no_nan = ~isnan(this_matrix_th(:,t));
        promedio_th(t) =  trimmean(this_matrix_th(no_nan, t), trim_range);
    end
    promedio_th = movmean(promedio_th, mov_mean_wind)/psth_bin;
    plot(onset_time,promedio_th, 'b')
    hold on
    promedio_bh = nan(1,size(this_matrix_beh,2));
    for t=1:size(this_matrix_beh,2)
        no_nan = ~isnan(this_matrix_beh(:,t));
        promedio_bh(t) =  trimmean(this_matrix_beh(no_nan, t), trim_range);
    end
      promedio_bh = movmean( promedio_bh, mov_mean_wind)/psth_bin;
    plot(onset_time, promedio_bh, 'r')
    xlim([-min_prev_dist 2*min_length])
    y_lim = ylim;
    plot([0 0], y_lim, 'k', 'HandleVisibility','off')
    title(num2str([100*nr/numel(sub_calls) numel(thicklish_indexes)]))
    % yticklabels([])
    pause(.1)
        sp = sp+1;

end