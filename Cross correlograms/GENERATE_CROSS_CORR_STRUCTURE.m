function phase_struct = GENERATE_CROSS_CORR_STRUCTURE(npx_data_dir, bin_size, psth_range )


% synch_directory     = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Synch data';
synch_directory     = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Synch data';
area2analyse        = 'PAG';
area_limit_table    = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\Area_limits_GoodLooking.xlsx';
behavior_data       = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Behavior backups';
play_behaviors      = {'Pounce', 'CC','Boxing', 'Evasion','Pin', 'Escape', 'CB', 'CD'};
non_play_behaviors  = {'Grooming', 'PounceI','Rearing', 'Sniffing','Scratching', 'Bite'};

% npx_raw_data = 
animal_code         = strsplit(npx_data_dir, '\');
animal_code         = animal_code{end};
animal_code_params  = strsplit(animal_code, ' ');
animal_batch        = animal_code_params{1};
repeated_animal     = animal_code_params{3};

area2analyze = 'PAG';
%% define parameters
n_rand              = 1000;

edges_freq = psth_range_freq(1):bin_size:psth_range_freq(2);


%% load synch from synch folder
load([synch_directory,'\', animal_code, '\synch_model_video2NPX.mat'], 'synch_model_video2NPX')

%% 2 Load beahvior data from behavior folder


Behavior_file =[behavior_data,'\', animal_code,'.txt'];%load behavior data

Behavior                            = readtable(Behavior_file);
Behavior(:,2)                       = [];
Behavior.Properties.VariableNames   = {'Animal', 'Start', 'End', 'Length', 'Type'};


bin_size                = 0.01;
conv_length             = 1;
Behavior.Type2          = Behavior.Type;

Behavior.Type2(ismember(Behavior.Type2, {'Pounce_A', 'Pounce_B'}))      = {'Pounce'}; %% Merging behaviors to Type2
Behavior.Type2(ismember(Behavior.Type2, {'Pounce_Ai', 'Pounce_Bi'}))    = {'PounceI'};
Behavior.Type2(ismember( Behavior.Type2,''))                            = {'Other'};
Behavior(ismember(Behavior.Animal, 'Reversal'),:)                       = [];

Behavior.Start          = predict(synch_model_video2NPX, Behavior.Start);
Behavior.End            = predict(synch_model_video2NPX, Behavior.End);


partner_sessions = Behavior(strcmp(Behavior.Type2, 'Partners session'),:);



%% 2 Create play bout array

bin_size                = 0.01;
conv_length             = 1;
animal_types            = unique(Behavior.Animal);

animal_types(ismember(animal_types,'Session_structure'))                =[];




config.Behavior         = Behavior;
config.repeated_animal  = repeated_animal;
config.animal_types     = animal_types        ;
config.play_behaviors   = play_behaviors      ;
config.beh_bin          = bin_size             ;
config.conv_length      = conv_length;
config.behavior_window  = 0;


[play_bouts_table]      = play_bout(config);

pre_play_bouts = build_balanced_periods(play_bouts_table);


config.Behavior         = Behavior;
config.repeated_animal  = repeated_animal;
config.animal_types     = animal_types        ;
config.play_behaviors   = non_play_behaviors      ;
config.beh_bin          = bin_size             ;
config.conv_length      = conv_length;
config.behavior_window  = 0;

[non_play_bouts_table]      = play_bout(config);
pre_non_play_bouts          = build_balanced_periods(non_play_bouts_table);


%% determining NPX type
hard_coded_x_coords_NPX2 = [8 40;258 290; 508 540; 758 790];
load([npx_data_dir,'\','chann_map_', area2analyse, '.mat'], 'chanMap', 'xcoords', 'ycoords')
if any(ismember(xcoords, hard_coded_x_coords_NPX2))
     NPX_Type        = 2;
else
     NPX_Type        = 1;
     if ~ismember(192, chanMap)

         pos_191 = find(chanMap==191);
         pos_193 = find(chanMap==193);

         if pos_193 == pos_191+1

             xcoords = [xcoords;NaN];
             xcoords(pos_193+1:end) = xcoords(pos_193:end-1);
             xcoords(pos_193) = 43;
             ycoords = [ycoords;NaN];
             ycoords(pos_193+1:end) = ycoords(pos_193:end-1);
             ycoords(pos_193) = 1900;
             chanMap = [chanMap;NaN];
             chanMap(pos_193+1:end) = chanMap(pos_193:end-1);
             chanMap(pos_193) = 192;
         else
             disp('Inconsistent ChannelMap')
             return
         end
     end
end

%% Create (load) channel map
disp('Loading Channel Map')
areas_by_channel = cell(384,1);
channel_map      = nan(384,2);

area_limit = readtable(area_limit_table);

% Build animal identifier for area selection
if strcmp(repeated_animal, 'Single2')
    this_animal = ['Batch', animal_batch(2), repeated_animal];
else
    this_animal = ['Batch', animal_batch(2), repeated_animal,animal_batch(4)];
end
area_limit = area_limit(ismember(area_limit.AnimalName,this_animal),:);

if NPX_Type == 1

    
  for ch_n=1:384
      ch = chanMap(ch_n);
      channel_map(ch,1) = xcoords(ch_n);
      channel_map(ch,2) = ycoords(ch_n);
      areas_by_channel{ch} = area_limit.area{ycoords(ch_n)>=area_limit.depth_start &  ycoords(ch_n)<area_limit.depth_end+1 & ismember(area_limit.Probe_Area, area2analyse) };
  end
else

  
    
   for ch_n=1:384
      probe_n = find(any(ismember(hard_coded_x_coords_NPX2,xcoords(ch_n)),2));
      ch = chanMap(ch_n);
      channel_map(ch,1) = xcoords(ch_n);
      channel_map(ch,2) = ycoords(ch_n);
      areas_by_channel{ch} = area_limit.area{ycoords(ch_n)>=area_limit.depth_start &  ycoords(ch_n)<area_limit.depth_end+1 & area_limit.ProbeNum==probe_n & ismember(area_limit.Probe_Area, 'PAG')};
  end

end   


%% load spikes
spike_times             = double(readNPY([npx_data_dir,'\spike_times_', area2analyze, '.npy']))/30000;
spike_clusters          = readNPY([npx_data_dir,'\spike_clusters_', area2analyze, '.npy']);
cluster_info            = readtable([npx_data_dir,'\cluster_info_', area2analyze, '.tsv'] ,"FileType","text",'Delimiter', '\t');
cluster_info            = cluster_info(ismember(cluster_info.group,{'mua', 'good'}),:);
these_neurons_areas     = areas_by_channel(cluster_info.ch+1);
cluster_info.area       = these_neurons_areas;
cluster_info.ch         = cluster_info.ch+1;
channels_with_spikes    = unique(cluster_info.ch)';



%% obtain_psth



PAG_LFP         = double(LFP);
clear LFP
sr_LFP = 2500;


% --- Step 1: pre alocated arrays ---
lfp_time                        = (1:size(PAG_LFP,2))/sr_LFP;
session_phase_stats             = nan(size(partner_sessions,1), size(cluster_info,1),7);
entire_recording_phase_stats    = nan(1, size(cluster_info,1),7);
play_phase_stats                = nan(1, size(cluster_info,1),7);
pre_play_phase_stats            = nan(1, size(cluster_info,1),7);
non_play_phase_stats            = nan(1, size(cluster_info,1),7);
pre_non_play_phase_stats        = nan(1, size(cluster_info,1),7);

session_psth                    = nan(size(partner_sessions,1), size(cluster_info,1),numel(edges_freq)-1);
entire_recording_psth           = nan(1, size(cluster_info,1),numel(edges_freq)-1);
play_psth                       = nan(size(partner_sessions,1), size(cluster_info,1),numel(edges_freq)-1);
pre_play_psth                   = nan(1, size(cluster_info,1),numel(edges_freq)-1);
non_play_psth                    = nan(size(partner_sessions,1), size(cluster_info,1),numel(edges_freq)-1);
pre_non_play_psth               = nan(1, size(cluster_info,1),numel(edges_freq)-1);
assigned                        = false( size(cluster_info,1),1);
end