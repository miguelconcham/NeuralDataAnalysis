function [psth_tensor_onset, psth_tensor_offset,psth_edges_onset,psth_edges_offset, mean_rate_during_call, all_call_index, cluster_area, call_length, CallStats, call_type,calls_within,psth_calls_onset,behavior_type,behavior_data,psth_edges] = get_psth_calls(current_dir,behaviors2analyse,area2analyze,range_borders,psth_bin)


% range_borders     = [-1 1];
% psth_bin          = 0.01;


%%
call_directory       = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\CallDetectionBackup';
behavior_files      = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Behavior backups';
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


disp('Loading Data')
Call_file   = [call_directory,'\',animal_code, '_Stats.xlsx']
Behavior_file   = [behavior_files,'\',animal_code, '.txt']


CallStats                           = readtable(Call_file);
CallStats.Properties.VariableNames  = cellfun(@(x) strrep(x, '_', ''),CallStats.Properties.VariableNames, 'UniformOutput',false );
CallStats.BeginTimes                = predict(synch_model_audio2NPX, CallStats.BeginTimes);
CallStats.EndTimes                  = predict(synch_model_audio2NPX, CallStats.EndTimes);
Behavior                            = readtable(Behavior_file);
Behavior(:,2)                       = [];
Behavior.Properties.VariableNames   = {'Animal', 'Start', 'End', 'Length', 'Type'};
Partner_sessions  = Behavior{ismember(Behavior.Type,'Partners session'),{'Start','End'}};


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

%% load spikes
cd(current_dir)
area_limit = readtable(area_limit_table);
area_limit.depth_start  = floor(area_limit.depth_start);
area_limit.depth_end    = ceil(area_limit.depth_end);
spike_times     = double(readNPY(['spike_times_', area2analyze, '.npy']))/30000;
spike_clusters  = readNPY(['spike_clusters_', area2analyze, '.npy']);
cluster_info    = readtable(['cluster_info_', area2analyze, '.tsv'] ,"FileType","text",'Delimiter', '\t');
cluster_info    = cluster_info(ismember(cluster_info.group,{'mua', 'good'}),:);
cluster_channels =cluster_info.ch;
load(['chann_map_', area2analyze, '.mat'], 'xcoords','ycoords','chanMap')
 hard_coded_x_coords = [8 40;258 290; 508 540; 758 790];
cluster_area = cell(size(cluster_info,1),1);
if strcmp(animal_batch(3), 'D')
    animal_type = 'Dual';
else
    animal_type = 'Single';
end

this_animal_id = ['Batch', animal_batch(2), animal_type, animal_batch(4)];

for j=1:numel(cluster_area)
    if any(any(ismember(hard_coded_x_coords,xcoords ),2))
        this_ch_xpos =  xcoords(ismember(chanMap,cluster_channels(j)));

        probe_n = find(any(ismember(hard_coded_x_coords,this_ch_xpos),2));
        cluster_area(j) = area_limit.area(area_limit.depth_start<cluster_info.depth(j) & area_limit.depth_end>=cluster_info.depth(j) & ismember(area_limit.AnimalName, this_animal_id) & area_limit.ProbeNum==probe_n & ismember(area_limit.Probe_Area, area2analyze));
    else
        cluster_area(j) = area_limit.area(area_limit.depth_start<cluster_info.depth(j) & area_limit.depth_end>=cluster_info.depth(j) & ismember(area_limit.AnimalName, this_animal_id) & ismember(area_limit.Probe_Area, area2analyze));
    end
end


%% plot calls these behaviors





beh_indexes = find(ismember(Behavior.Type, behaviors2analyse) & ismember(Behavior.Animal, repeated_animal));
[ordered_beh_length,beh_length_order] = sort(Behavior.End(beh_indexes) - Behavior.Start(beh_indexes));
beh_indexes = beh_indexes(beh_length_order);

figure
subplot(3,1,[1 2])
hold on

% here are hard coded the bin edges for the calls
behavior_call_range = [-20 20];
behavior_call_bin = 0.01;
psth_edges = behavior_call_range(1):behavior_call_bin:behavior_call_range(2);
psth_calls_onset = zeros(numel(beh_indexes),numel(psth_edges));
calls_within = [];
behavior_type = {};
for j=1:numel(beh_indexes)
    beh_start = Behavior.Start(beh_indexes(j));
    beh_end   = Behavior.End(beh_indexes(j));
    this_beh_length = beh_end-beh_start;
    this_behavior_type = Behavior.Type{beh_indexes(j)};

    these_calls = find(CallStats.BeginTimes>=beh_start+behavior_call_range(1) & CallStats.BeginTimes<=beh_start+this_beh_length+behavior_call_range(2));

    for cn = 1:numel(these_calls)
        call_beg  = CallStats.BeginTimes(these_calls(cn));
        call_end  = CallStats.EndTimes(these_calls(cn));

        if (call_end>=beh_start && call_end<=beh_end) || ...
                (call_beg>=beh_start && call_beg<=beh_end)
            calls_within = [calls_within;[these_calls(cn), this_beh_length]];
            behavior_type = [behavior_type,this_behavior_type];
        end

           
        psth_calls_onset(j,psth_edges>=call_beg-beh_start & psth_edges<=call_end-beh_start) = 1;       

        fill([call_beg call_end call_end  call_beg]-beh_start, [-.5 -.5 .5 .5]+j, 'r', 'EdgeColor','None')
    end
end
hold on
plot(ordered_beh_length,1:numel(beh_indexes), 'k')
plot(ordered_beh_length*0,1:numel(beh_indexes), 'k')
axis([-5 10 .5 numel(beh_indexes)+.5])
title([behaviors2analyse{:}])


subplot(3,1,3)
plot(psth_edges, movmean(mean(psth_calls_onset),10), 'r')

y_lim = ylim;
hold on
plot([0 0 ], y_lim, 'k')



%%

Partner_sessions    = Behavior{ismember(Behavior.Type,'Partners session'),{'Start','End'}};
ThihklingSession    = Behavior{ismember(Behavior.Type,'Tickling'),{'Start','End'}};
call_indexes_play_sessions = find(any(CallStats.BeginTimes>=Partner_sessions(:,1)' & CallStats.EndTimes<=Partner_sessions(:,2)',2));
if isempty(ThihklingSession)
    ThihklingSession = [Inf Inf];
end
call_index_thikling = find(CallStats.BeginTimes>=ThihklingSession(1) & CallStats.EndTimes<=ThihklingSession(2));

call_type = repmat({'PlaySession'},numel(call_indexes_play_sessions),1 );

call_type(ismember(call_indexes_play_sessions,calls_within(:,1))) = {'DuringBehavior'};
call_type = [call_type;repmat({'Thikling'}, numel(call_index_thikling),1)];

all_call_index = [call_indexes_play_sessions;call_index_thikling];

%%

longest_call = max(CallStats.EndTimes(all_call_index)- CallStats.BeginTimes(all_call_index));
psth_edges_onset    = range_borders(1):psth_bin:ceil(range_borders(2)/psth_bin + longest_call/psth_bin)*psth_bin;
psth_edges_offset   = -ceil(range_borders(1)/psth_bin + longest_call/psth_bin)*psth_bin:psth_bin:range_borders(2);


psth_tensor_onset  = nan(size(cluster_info,1),numel(all_call_index), numel(psth_edges_onset)-1);
psth_tensor_offset = nan(size(cluster_info,1),numel(all_call_index), numel(psth_edges_offset)-1);
mean_rate_during_call = nan(size(cluster_info,1),numel(all_call_index),2);
normalized_rate_during_call = nan(size(cluster_info,1),numel(all_call_index));

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
    for bn = 1:numel(all_call_index)

        call_start   = CallStats.BeginTimes(all_call_index(bn));
        call_end     = CallStats.EndTimes(all_call_index(bn));
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

call_length = CallStats.EndTimes(all_call_index) - CallStats.BeginTimes(all_call_index);
behavior_data = [num2cell(beh_indexes) Behavior.Type(beh_indexes) num2cell(ordered_beh_length)];
behavior_data = cell2table(behavior_data);
behavior_data.Properties.VariableNames = {'IndexInBehavior', 'BehaviorType', 'BehaviorLenght'};
end





