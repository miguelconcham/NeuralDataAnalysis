function [psth_tensor_onset, psth_tensor_offset,psth_edges_onset,psth_edges_offset, mean_rate_during_behavior, behavior_indexes, cluster_area, pounce_length,Behavior,ego_alter_dummy, repeated_animal,cluster_info] = get_psth(current_dir,behaviors2analyse,area2analyze,range_borders,psth_bin)


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
% area_limit_axu = area_limit(ismember(area_limit.AnimalName, this_animal_id) & area_limit.ProbeNum==probe_n & ismember(area_limit.Probe_Area, area2analyze),:);
for j=1:numel(cluster_area)
    if any(any(ismember(hard_coded_x_coords,xcoords ),2))
        this_ch_xpos =  xcoords(ismember(chanMap,cluster_channels(j)));

        probe_n = find(any(ismember(hard_coded_x_coords,this_ch_xpos),2));
        cluster_area(j) = area_limit.area(area_limit.depth_start<cluster_info.depth(j) & area_limit.depth_end>=cluster_info.depth(j) & ismember(area_limit.AnimalName, this_animal_id) & area_limit.ProbeNum==probe_n & ismember(area_limit.Probe_Area, area2analyze));
    else
        cluster_area(j) = area_limit.area(area_limit.depth_start<cluster_info.depth(j) & area_limit.depth_end>=cluster_info.depth(j) & ismember(area_limit.AnimalName, this_animal_id) & ismember(area_limit.Probe_Area, area2analyze));
    end
end


%%

behavior_indexes = find(ismember(Behavior.Type, behaviors2analyse) );
ego_alter_dummy = false(size(behavior_indexes));
ego_alter_dummy(ismember(Behavior.Animal(behavior_indexes), repeated_animal)) = true;

longest_behavior = max(Behavior.End(behavior_indexes) - Behavior.Start(behavior_indexes));

psth_edges_onset    = range_borders(1):psth_bin:ceil(range_borders(2) + longest_behavior/psth_bin)*psth_bin;
psth_edges_offset   = -ceil(range_borders(1) + longest_behavior/psth_bin)*psth_bin:psth_bin:range_borders(2);


psth_tensor_onset  = nan(size(cluster_info,1),numel(behavior_indexes), numel(psth_edges_onset)-1);
psth_tensor_offset = nan(size(cluster_info,1),numel(behavior_indexes), numel(psth_edges_offset)-1);
mean_rate_during_behavior = nan(size(cluster_info,1),numel(behavior_indexes),2);
normalized_rate_during_behavior = nan(size(cluster_info,1),numel(behavior_indexes));

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
    for bn = 1:numel(behavior_indexes)

        beh_start   = Behavior.Start(behavior_indexes(bn));
        beh_end     = Behavior.End(behavior_indexes(bn));
        mean_rate_during_behavior(cn,bn,1) = sum(this_neuron_spikes>=beh_start & this_neuron_spikes<=beh_end)/(beh_end-beh_start);

         mean_rate_during_behavior(cn,bn,2) = sum(this_neuron_spikes>=beh_start-5 & this_neuron_spikes<=beh_start)/5;
        if  mean_rate_during_behavior(cn,bn,1)>max(rate_val)
            normalized_rate_during_behavior(cn,bn)  = 1;
        elseif mean_rate_during_behavior(cn,bn,1)<min(rate_val)
            normalized_rate_during_behavior(cn,bn) = 0;
        else
            normalized_rate_during_behavior(cn,bn) = rate_count(rate_val == floor(mean_rate_during_behavior(cn,bn,1)) );
        end
        this_behavior_spikes = this_neuron_spikes(this_neuron_spikes>=beh_start+range_borders(1) & this_neuron_spikes<=beh_end+range_borders(2));
        psth_tensor_onset(cn,bn,:) = histcounts(this_behavior_spikes-beh_start,psth_edges_onset );
        psth_tensor_onset(cn,bn,psth_edges_onset(2:end)>beh_end-beh_start+range_borders(2))=NaN;



          psth_tensor_offset(cn,bn,:) = histcounts(this_behavior_spikes-beh_end,psth_edges_offset );
        psth_tensor_offset(cn,bn,psth_edges_offset(1:end-1)<beh_start-beh_end-range_borders(1))=NaN;


    end
   

end

pounce_length = Behavior.End(behavior_indexes) - Behavior.Start(behavior_indexes);
end





