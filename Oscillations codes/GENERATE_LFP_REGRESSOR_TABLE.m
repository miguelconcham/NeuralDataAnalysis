function stack_lm_pow_structure = GENERATE_LFP_REGRESSOR_TABLE(current_dir)

hmm_directory       = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\HMM data\HMM raw data';
synch_directory     = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Synch data';
area_limit_table    = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\Area_limits_GoodLooking.xlsx';
call_directory      = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\CallDetectionBackup';
tracking_data       = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Traking backups';
behavior_data       = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Behavior backups';
cd(current_dir)
animal_code         = strsplit(current_dir, '\');
animal_code         = animal_code{end};
animal_code_params  = strsplit(animal_code, ' ');
animal_batch        = animal_code_params{1};
date                = animal_code_params{2};
repeated_animal     = animal_code_params{3};

%% load synch from synch folder
cd([synch_directory, '\', animal_code])
load([synch_directory,'\', animal_code, '\synch_model_video2NPX.mat'])
load([synch_directory,'\', animal_code, '\synch_model_audio2NPX.mat'])
current_dir = cd;
play_behaviors      = {'Pounce', 'CC','Boxing', 'Evasion','Pin', 'Escape', 'CB', 'CD'};

%% 2 Load hmm data from HMM folder

pt = 1;
load([hmm_directory, '\',animal_code,' P', num2str(pt), ' binned time.mat'])
hmm_binned_time = adjusted_binned_time;
bin_size = mean(diff(hmm_binned_time));
load([tracking_data,'\',animal_code,' P', num2str(pt), ' traking_structure.mat']) 
restrict2Partnerssession = true;

disp('Loading Data')
Call_file   = [call_directory,'\', animal_code,'_Stats.xlsx' ]; %load call data
Behavior_file =[behavior_data,'\', animal_code,'.txt'];%load behavior data

CallStats                           = readtable(Call_file);
CallStats.Properties.VariableNames  = cellfun(@(x) strrep(x, '_', ''),CallStats.Properties.VariableNames, 'UniformOutput',false );
CallStats.BeginTimes                = predict(synch_model_audio2NPX, CallStats.BeginTimes);
CallStats.EndTimes                  = predict(synch_model_audio2NPX, CallStats.EndTimes);
Behavior                            = readtable(Behavior_file);
Behavior(:,2)                       = [];
Behavior.Properties.VariableNames   = {'Animal', 'Start', 'End', 'Length', 'Type'};
Behavior(ismember(Behavior.Type,'Partners session'),:)


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

partner_names = animal_types;
partner_names(ismember(animal_types, repeated_animal)) = [];

config.Behavior         = Behavior;
config.repeated_animal  = repeated_animal;
config.animal_types     = animal_types        ;
config.play_behaviors   = play_behaviors      ;
config.beh_bin          = beh_bin             ;
config.conv_length      = conv_length;
config.behavior_window  = 0;

traking_time        = predict(synch_model_video2NPX, (traking_structure.frames2stract/30)')';
animal_pos          = nan(numel(hmm_binned_time),2);
partner_pos         = nan(numel(hmm_binned_time),2);
animal_pos(:,1)     = interp1(traking_time, smoothdata(traking_structure.animal_pos(:,1), 'loess',5), hmm_binned_time,'cubic');
animal_pos(:,2)     = interp1(traking_time,smoothdata(traking_structure.animal_pos(:,2), 'loess',5), hmm_binned_time,'cubic');


partner_pos(:,1) = interp1(traking_time, smoothdata(traking_structure.partner_pos(:,1), 'loess',5), hmm_binned_time,'cubic');
partner_pos(:,2) = interp1(traking_time,smoothdata(traking_structure.partner_pos(:,2), 'loess',5), hmm_binned_time,'cubic');




[play_bouts_table]  = play_bout(config);
hmm_binned_time     = predict(synch_model_audio2NPX,hmm_binned_time')';
play_bout_time      = any(hmm_binned_time>=play_bouts_table(:,1) & hmm_binned_time<=play_bouts_table(:,2))';
min_time2analysis   = Behavior.Start(ismember(Behavior.Type,'Partners session'))  ;
max_time2analysis   = Behavior.End(ismember(Behavior.Type,'Partners session'))  ;

time_limit          = (hmm_binned_time>= min_time2analysis(pt) & hmm_binned_time<=  max_time2analysis(pt) )';
Behavior.Start      = predict(synch_model_audio2NPX    , Behavior.Start);
Behavior.End        = predict(synch_model_audio2NPX, Behavior.End);

hmm_states =  readNPY( [hmm_directory, '\',animal_code,' P', num2str(pt),'_states_K2.npy']);

play_state       = 0;
non_play_state      = 1;

classificator_tp = sum(ismember(hmm_states(time_limit),play_state) & play_bout_time((time_limit)) ==1);
classificator_tn = sum(ismember(hmm_states(time_limit),non_play_state) & play_bout_time((time_limit)) ==0);
classificator_fn = sum(ismember(hmm_states(time_limit),non_play_state) & play_bout_time((time_limit)) ==1);
classificator_fp = sum(ismember(hmm_states(time_limit),play_state) & play_bout_time((time_limit))==0);

if classificator_tp<classificator_fn
    play_state=1;
    non_play_state =0;
    classificator_tp = sum(ismember(hmm_states(time_limit),play_state) & play_bout_time((time_limit))==1);
    classificator_tn = sum(ismember(hmm_states(time_limit),non_play_state) & play_bout_time((time_limit)) ==0);
    classificator_fn = sum(ismember(hmm_states(time_limit),non_play_state) & play_bout_time((time_limit)) ==1);
    classificator_fp = sum(ismember(hmm_states(time_limit),play_state) & play_bout_time((time_limit)) ==0);
end

figure
bar([([classificator_tp classificator_fn])/(classificator_fn+classificator_tp) [classificator_tn classificator_fp]/(classificator_tn+ classificator_fp)])
xticklabels({'TruePositives','FalseNegative','TrueNegative','FalsePositive'})
pause(.1)

hmm_states = hmm_states==play_state;

if hmm_states(end)==1
    end_state= [find(diff(hmm_states)==-1);numel(hmm_states)];
else
    end_state= find(diff(hmm_states)==-1);
end
if hmm_states(1)==1
    start_state= [1;find(diff(hmm_states)==1)];
else
    start_state= find(diff(hmm_states)==1);
end
beg_end_times = hmm_binned_time([start_state end_state  ]);




%%  load lfp from current dir
disp('LOADING LFP')
if exist([current_dir,'\','LFP_PAG.mat'], 'file')==2

    NPX_Type        = 2;
    load LFP_PAG
elseif exist([current_dir,'\','LFP_PAG.dat'], 'file')==2
    NPX_Type        = 1;
    file_pointer    = fopen('LFP_PAG.dat', 'r');
    LFP             = fread(file_pointer,'int16');
    LFP             = reshape(LFP, 384, numel(LFP)/384);
end


disp('LFP LOADED')
%% select_mid_pag_channel
disp('Loading Channel Map')
hard_coded_x_coords = [8 40;258 290; 508 540; 758 790];
area_limit = readtable(area_limit_table);
if strcmp(repeated_animal, 'Single2')
    this_animal = ['Batch', animal_batch(2), repeated_animal];
else
this_animal = ['Batch', animal_batch(2), repeated_animal,animal_batch(4)];
end
area_limit = area_limit(ismember(area_limit.AnimalName,this_animal),:);

if NPX_Type == 1

    PAG_channels = area_limit{ismember(area_limit.area, {'LPAG'}), {'ch_start', 'ch_end'}};
    PAG_channels = str2double(PAG_channels);
    channel_Range = [min(PAG_channels(:)) max(PAG_channels(:))];
    mid_PAG_channel = round(mean(channel_Range));
else
    load ChannelMap
    Y_Range = area_limit{ismember(area_limit.area, {'LPAG'}), {'ProbeNum','depth_start', 'depth_end'}};
    mid_PAG_channel = nan(size(Y_Range,1),1);
    figure
    plot(xcoords,ycoords, 'k.')
    hold on


    for j=1:size(Y_Range,1)
        this_indexes = ycoords>=Y_Range(j,2) & ycoords<=Y_Range(j,3) & ismember(xcoords,hard_coded_x_coords(Y_Range(j,1),:));

        all_locs = [xcoords(this_indexes) ycoords(this_indexes)];
        plot(all_locs(:,1),all_locs(:,2), 'r.')

        mean_loc = mean(all_locs);
        [~, closest_channel]= min(sum(abs([xcoords ycoords]-repmat(mean_loc,numel(ycoords),1)),2));
        plot(xcoords(closest_channel), ycoords(closest_channel), 'xb')
        mid_PAG_channel(j) = chanMap(closest_channel);

    end

end
%% obtain_psth
hist_range = [-20 20];
play_bouts_this_partner = play_bouts_table(play_bouts_table(:,1)>=min_time2analysis(pt)+hist_range(1) & play_bouts_table(:,2)<=max_time2analysis(pt)+hist_range(2),:);

hist_edges = hist_range(1):bin_size:hist_range(2);
range_time_wrap = [-5 5];
pre_time_edges  = range_time_wrap(1):bin_size:0;
post_time_edges = 0:bin_size: range_time_wrap(2);
n_bins_time_wrap = range(range_time_wrap)*10;;

PAG_LFP         = double(LFP(mid_PAG_channel,:));
clear LFP
LFP_time        = (1:size(PAG_LFP,2))/2500;
wind_length = .250;
wind_overlap = 0.240;
f = 4:.1:15;
theta_range = [6 12];

f_index = f>=theta_range(1) & f<=theta_range(2);
range2exctract                  = LFP_time>=min_time2analysis(pt)+hist_range(1) & LFP_time<=max_time2analysis(pt)+hist_range(2);
for ch_n=1:numel(mid_PAG_channel)
    disp('Estimating spectrogram')
    [pow_spectrogram,~,spect_time]  = spectrogram(PAG_LFP(ch_n,range2exctract),wind_length*2500, wind_overlap*2500, f,2500);
    spect_time                     = spect_time+min_time2analysis(pt)+hist_range(1);

    pow_spectrogram = abs(pow_spectrogram);
    disp('ready')

    signal_in_range      = zscore(mean(log10(pow_spectrogram)))<3;

    theta_pow_spect_t =  mean(log10(pow_spectrogram(f_index,:)));
    theta_pow_spect_t   = movmean(theta_pow_spect_t,1/max(theta_range));

    theta_pow          =  mean(log10(pow_spectrogram(f>=theta_range(1) & f<=theta_range(2),:)));
    theta_pow           = movmean(theta_pow,1/max(theta_range));
    theta_pow           = interp1(spect_time,theta_pow,hmm_binned_time);


    play_bout_onset_this_pt     = nan(size(play_bouts_this_partner,1), numel(hist_edges));
    play_bout_offset_this_pt    = nan(size(play_bouts_this_partner,1), numel(hist_edges));
    play_bout_tw_this_pt        = nan(size(play_bouts_this_partner,1), numel(pre_time_edges)+numel(post_time_edges)-2 + n_bins_time_wrap);



    for pb_n=1:size(play_bouts_this_partner,1)

        play_bout_start = play_bouts_this_partner(pb_n,1);
        play_bout_end   = play_bouts_this_partner(pb_n,2);


        [~,loc_start] =  min(abs(hmm_binned_time-play_bout_start));
        [~,loc_end] =  min(abs(hmm_binned_time-play_bout_end));

        entire_range = round(loc_start+(hist_range(1)/bin_size):loc_start+(hist_range(2)/bin_size));
        allowed_index_theta = ismember(entire_range,1:size(theta_pow,2));

        play_bout_onset_this_pt(pb_n,allowed_index_theta) = theta_pow(entire_range(allowed_index_theta));


        entire_range = round(loc_end+(hist_range(1)/bin_size):loc_end+(hist_range(2)/bin_size));
        allowed_index_theta = ismember(entire_range,1:size(theta_pow,2));
        play_bout_offset_this_pt(pb_n,allowed_index_theta) = theta_pow(entire_range(allowed_index_theta));



        pre_time    = loc_start+ round(range_time_wrap(1)/bin_size):loc_start-1;
        post_time   = loc_end+1:loc_end+round(range_time_wrap(2)/bin_size);

        in_between_time = loc_start:loc_end

        time_wrapped_theta = interp1(in_between_time, theta_pow(in_between_time), linspace(loc_start,loc_end,n_bins_time_wrap))

        all_wraped = [theta_pow(pre_time),time_wrapped_theta, theta_pow(post_time)];


    end
    
    
    
    
    
    
    
    

end
end
