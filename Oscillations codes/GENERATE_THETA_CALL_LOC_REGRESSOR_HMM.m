function psth_struct = GENERATE_THETA_CALL_LOC_REGRESSOR_HMM(animal_code, pt, f, freq_range)
% GENERATE_THETA_PSTH
% Computes freq-band (6–12 Hz) LFP power peri-event time histograms (PSTHs)
% around social play bouts and key behaviors. Outputs a struct with aligned data.

varNames2store = { ...
    'state_onset', 'state_offset', ...
    'state_onset_regressor', 'state_offset_regressor', ...
    'play_bout_onset_regressor', 'play_bout_offset_regressor', ...
    'call_onset_regressor', 'call_offset_regressor', ...
    'self_speed_onset_regressor', 'self_speed_offset_regressor', ...
    'self_acc_onset_regressor', 'self_acc_offset_regressor', ...
    'other_speed_onset_regressor', 'other_speed_offset_regressor', ...
    'other_acc_onset_regressor', 'other_acc_offset_regressor', ...
    'animal_distance_onset_regressor', 'animal_distance_offset_regressor', ...
    'self_onset_regressor', 'self_offset_regressor', ...
    'other_onset_regressor', 'other_offset_regressor', ...
    'time_wrapped','wrapped_bins','play_bout_latency',...
    'hist_range', 'wind_length', 'wind_overlap', ...
    'f', 'freq_range', 'mid_PAG_channel' ...
     'play_behaviors', 'CallStats', 'Behavior' ...
     'play_bouts_table', 'hmm_onset_offset','hmm_type' };
%% ---- DEFINE PARAMETERS FOR SPECTROGRAM

% Spectrogram parameters
hist_range      = [-20 20];       % peri-event window for onset/offset (s)


% Spectrogram parameters
wind_length     = .250;    % 250 ms
wind_overlap    = 0.245;   % 249 ms overlap
spect_bin_size  = wind_length-wind_overlap;

% f          = 4:.1:15;      % frequency range for spectrogram
% freq_range = [6 12];       % freq band

wrapped_bins = 100;

%% -------------------- SETUP AND PATHS --------------------
% Define behavior types for playbout 
play_behaviors = {'Pounce', 'CC','Boxing', 'Evasion','Pin', 'Escape', 'CB', 'CD'};

% Define data directories for loading data
call_folder         =  '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\CallDetectionBackup';
synch_directory     = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Synch data';
area_limit_table    = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\Area_limits_GoodLooking.xlsx';
npx_folder          = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\NPX data\NPX raw data';
trakcking_folder    = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Traking backups';
behavior_data       = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Behavior backups';
hmm_data_folder     = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\HMM 2 and 3 states 2 partners' ;
% Extract animal/session code information from animal_code

animal_code_params = strsplit(animal_code, ' ');
animal_batch       = animal_code_params{1};
date               = animal_code_params{2};
repeated_animal    = animal_code_params{3};

%% -------------------- LOAD SYNCHRONIZATION MODEL --------------------
% Load mapping between video time and neural time
load([synch_directory,'\', animal_code, '\synch_model_video2NPX.mat'])
load([synch_directory,'\', animal_code, '\synch_model_audio2NPX.mat'])

%% --------------------------- LOAD TRAKING AND ANIMAL VARIABLES for partner = pt --------


load([trakcking_folder,'\',animal_code,' P', num2str(pt), ' traking_structure.mat']) 


traking_time        = predict(synch_model_video2NPX, (traking_structure.frames2stract/30)')';
full_traking_time   = (traking_structure.frames2stract(1):traking_structure.frames2stract(end))/30;
full_traking_time   = predict(synch_model_video2NPX, full_traking_time')';
animal_pos          = nan(numel(full_traking_time),2);
partner_pos         = nan(numel(full_traking_time),2);
animal_pos(:,1)     = interp1(traking_time, smoothdata(traking_structure.animal_pos(:,1), 'loess',5), full_traking_time,'cubic');
animal_pos(:,2)     = interp1(traking_time,smoothdata(traking_structure.animal_pos(:,2), 'loess',5), full_traking_time,'cubic');


partner_pos(:,1)    = interp1(traking_time, smoothdata(traking_structure.partner_pos(:,1), 'loess',5), full_traking_time,'cubic');
partner_pos(:,2)    = interp1(traking_time,smoothdata(traking_structure.partner_pos(:,2), 'loess',5), full_traking_time,'cubic');


animal_dist         = sqrt(sum((partner_pos-animal_pos).*(partner_pos-animal_pos),2));
animal_speed        = sqrt(sum(diff(animal_pos).*diff(animal_pos),2));
animal_accel        = abs(diff(animal_speed));


partner_speed       = sqrt(sum(diff(partner_pos).*diff(partner_pos),2));
partner_accel       = abs(diff(partner_speed));

%% ----------- LOAD BEHAVIOR DATA AND OBTAIN PLAYBOUTS


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
Behavior(ismember(Behavior.Animal, 'Session_structure'),:)              = [];

animal_types                                                            = unique(Behavior.Animal);
disp('Animal this session')
disp(animal_types')


Behavior.Start          = predict(synch_model_video2NPX, Behavior.Start);
Behavior.End            = predict(synch_model_video2NPX, Behavior.End);

config.Behavior         = Behavior;
config.repeated_animal  = repeated_animal;
config.animal_types     = animal_types        ;
config.play_behaviors   = play_behaviors      ;
config.beh_bin          = bin_size             ;
config.conv_length      = conv_length;
config.behavior_window  = 0;


[play_bouts_table]      = play_bout(config);

%% Load HMM onset and offset


prediction_struct_files = dir([hmm_data_folder,'\',animal_code,' P', num2str(pt) ' prediction_struct*']);
load([hmm_data_folder,'\',prediction_struct_files.name],'prediction_struct')

prediction_struct.filled_play_bouts(:,1) = predict(synch_model_audio2NPX,prediction_struct.filled_play_bouts(:,1));
prediction_struct.filled_play_bouts(:,2) = predict(synch_model_audio2NPX,prediction_struct.filled_play_bouts(:,2));

hmm_onset_offset = prediction_struct.filled_play_bouts;
hmm_type = any(prediction_struct.is_this_hmm & prediction_struct.is_there_play_beh,2);

%% Load calls
CallStats = readtable([call_folder, '\', animal_code, '_Stats.xlsx']) ;
CallStats.Properties.VariableNames = cellfun(@(x) strrep(x, '_', ''),CallStats.Properties.VariableNames, 'UniformOutput',false );
CallStats.BeginTimes = predict(synch_model_audio2NPX,CallStats.BeginTimes);
CallStats.EndTimes = predict(synch_model_audio2NPX,CallStats.EndTimes);
%% -------------------- LOAD LFP DATA --------------------
disp('LOADING LFP')
if exist([npx_folder,'\',animal_code,'\LFP_PAG.mat'], 'file')==2
    % Preprocessed LFP file exists
    NPX_Type = 2;
    load([npx_folder,'\',animal_code,'\LFP_PAG.mat'], 'LFP')
elseif exist([npx_folder,'\',animal_code,'\LFP_PAG.dat'], 'file')==2
    % Load raw binary LFP file
    NPX_Type = 1;
    file_pointer = fopen([npx_folder,'\',animal_code,'\LFP_PAG.dat'], 'r');
    LFP = fread(file_pointer,'int16');
    LFP = reshape(LFP, 384, numel(LFP)/384);
end
disp('LFP LOADED')


%% -------------------- SELECT PAG CHANNEL(S) --------------------
disp('Loading Channel Map')
hard_coded_x_coords = [8 40;258 290; 508 540; 758 790];
area_limit = readtable(area_limit_table);

% Build animal identifier for area selection
if strcmp(repeated_animal, 'Single2')
    this_animal = ['Batch', animal_batch(2), repeated_animal];
else
    this_animal = ['Batch', animal_batch(2), repeated_animal,animal_batch(4)];
end
area_limit = area_limit(ismember(area_limit.AnimalName,this_animal),:);

if NPX_Type == 1
    % Raw LFP: select channel range for LPAG region
    PAG_channels = area_limit{ismember(area_limit.area, {'LPAG'}), {'ch_start', 'ch_end'}};
    PAG_channels = str2double(PAG_channels);
    channel_Range = [min(PAG_channels(:)) max(PAG_channels(:))];
    mid_PAG_channel = round(mean(channel_Range));
else
    % Preprocessed: use ChannelMap.mat to locate mid-PAG channel
    load([npx_folder,'\',animal_code,'\ChannelMap.mat'], 'xcoords', 'ycoords','chanMap')
    Y_Range = area_limit{ismember(area_limit.area, {'LPAG'}), {'ProbeNum','depth_start', 'depth_end'}};
    mid_PAG_channel = nan(size(Y_Range,1),1);
    figure
    plot(xcoords,ycoords, 'k.'); hold on
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

%% -------------------- COMPUTE SPECTROGRAM & THETA POWER --------------------

% Extract LFP from mid-PAG channel(s)
PAG_LFP = double(LFP(mid_PAG_channel,:));
clear LFP
LFP_time = (1:size(PAG_LFP,2))/2500; % sample rate: 2.5 kHz

% Identify freq indices in spectrogram
f_index = f>=freq_range(1) & f<=freq_range(2);

% Only compute spectrogram over LFP range covering all play bouts
range2exctract      = LFP_time>=min(traking_time)+hist_range(1) & LFP_time<=max(traking_time)+hist_range(2);
play_bouts_table    = play_bouts_table(play_bouts_table(:,1)>=min(traking_time) & play_bouts_table(:,2)<=max(traking_time),:);

slef_onset_offset = [Behavior.Start(ismember(Behavior.Animal,repeated_animal)) Behavior.End(ismember(Behavior.Animal,repeated_animal))];
other_onset_offset = [Behavior.Start(~ismember(Behavior.Animal,repeated_animal)) Behavior.End(~ismember(Behavior.Animal,repeated_animal))];

% Initialize output struct
psth_struct = [];
pre_index = 1:round(-hist_range(1)/spect_bin_size);
for ch_n=1:numel(mid_PAG_channel)
    disp('Estimating spectrogram')
    % Compute spectrogram for selected LFP channel
    [pow_spectrogram,~,spect_time]  = spectrogram(PAG_LFP(ch_n,range2exctract),wind_length*2500, floor(wind_overlap*2500), f,2500);
    spect_time = spect_time+min(traking_time)+hist_range(1) - .5*wind_length;
    disp('Estimating Onset and Offset PSTHs')
    animal_speed_spect_time     = interp1(full_traking_time(1:end-1)', animal_speed,spect_time');
    animal_accel_spect_time     = interp1(full_traking_time(2:end-1)',animal_accel,spect_time');
    partner_speed_spect_time    = interp1(full_traking_time(1:end-1)',partner_speed,spect_time');
    partner_accel_spect_time    = interp1(full_traking_time(2:end-1)',partner_accel,spect_time');
    animal_dist_spect_time    = interp1(full_traking_time,animal_dist,spect_time');
    pow_spectrogram = abs(pow_spectrogram);

    % Extract freq-band power (log-scaled, smoothed)
    freq_pow = mean(log10(pow_spectrogram(f_index,:)));
    freq_pow = movmean(freq_pow,1/max(freq_range));

    %% -------------------- COMPUTE PERI-EVENT THETA PSTHs --------------------
    % Allocate arrays for different peri-event alignments
    play_bout_latency                   = nan(size(hmm_onset_offset,1),2);
    state_onset                         = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));
    state_offset                        = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));
    time_wrapped                        = nan(size(hmm_onset_offset,1), round(-hist_range(1)/spect_bin_size)  +2*wrapped_bins ); 
   
    %Create all needed regressors (state, play bout, call, speed, acceleration,
    %self and other)
    state_onset_regressor                = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));
    state_offset_regressor               = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size) ); 

    play_bout_onset_regressor           = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));
    play_bout_offset_regressor          = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));
   
    call_onset_regressor                = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));
    call_offset_regressor               = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));

    self_speed_onset_regressor           = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));
    self_speed_offset_regressor         = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));
    self_acc_onset_regressor            = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));
    self_acc_offset_regressor           = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));

    other_speed_onset_regressor         = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));
    other_speed_offset_regressor        = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));
    other_acc_onset_regressor           = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));
    other_acc_offset_regressor          = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));

    self_onset_regressor                = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));
    self_offset_regressor               = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));

    other_onset_regressor               = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));
    other_offset_regressor              = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));

    animal_distance_onset_regressor     = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));    
    animal_distance_offset_regressor    = nan(size(hmm_onset_offset,1), round(range(hist_range)/spect_bin_size));

 
    % Loop over play bouts
    for hmm_n=1:size(hmm_onset_offset,1)
        % Get bout start/end
        hmm_start     = hmm_onset_offset(hmm_n,1);
        hmm_end       = hmm_onset_offset(hmm_n,2);
        
        [~,loc_start]       = min(abs(spect_time-hmm_start));
        [~,loc_end]         = min(abs(spect_time-hmm_end));     
        next_play_bout_start = min(play_bouts_table(play_bouts_table(:,1)>hmm_start,1));
        if isempty(next_play_bout_start)
            next_play_bout_start = Inf;
        end
        play_bout_latency(hmm_n,:) = [next_play_bout_start hmm_end]-hmm_start;

        if next_play_bout_start<hmm_end
            [~,loc_next_pb] = min(abs(spect_time-next_play_bout_start));
            index_between   = loc_start:loc_next_pb-1;
            if numel(index_between)>=2
                wrapped_between = interp1(index_between,freq_pow(index_between), linspace(loc_start,loc_next_pb-1,wrapped_bins));
            else
                wrapped_between = nan(1,wrapped_bins);
            end

            index_during    = loc_next_pb:loc_end;
            if numel(index_during)>=2
                wrapped_during  = interp1(index_during,freq_pow(index_during), linspace(loc_next_pb,loc_end,wrapped_bins));
            else
                wrapped_during = nan(1,wrapped_bins);
            end
        else
            index_between   =  loc_start:loc_end;
            wrapped_between = interp1(index_between,freq_pow(index_between), linspace(loc_start,loc_end,wrapped_bins));
            wrapped_during = nan(1,wrapped_bins);
        end



  

        % --- Align to state onset ---
        entire_range_onset                                      = round(loc_start+(hist_range(1)/spect_bin_size)):(loc_start+(hist_range(2)/spect_bin_size));
        if numel(entire_range_onset)>round(range(hist_range)/spect_bin_size)
            entire_range_onset(1) = [];
        end
        allowed_index_freq                                      = ismember(entire_range_onset,1:size(freq_pow,2));
        state_onset(hmm_n,allowed_index_freq)                   = freq_pow(entire_range_onset(allowed_index_freq));
        time_wrapped(hmm_n,pre_index)                           =  state_onset(hmm_n,pre_index);
        time_wrapped(hmm_n,pre_index(end)+1:pre_index(end)+wrapped_bins)  ...
                                                                =    wrapped_between;
        time_wrapped(hmm_n,pre_index(end)+wrapped_bins+1:pre_index(end)+2*wrapped_bins)  ...
                                                                =    wrapped_during;

        current_time_onset                                      = nan(1,size(call_onset_regressor,2));
        current_time_onset(allowed_index_freq)                  = spect_time(entire_range_onset(allowed_index_freq));       
        state_onset_regressor(hmm_n,:)                           = any(current_time_onset>=hmm_onset_offset(:,1) & current_time_onset<=hmm_onset_offset(:,2),1);       
        play_bout_onset_regressor(hmm_n,:)                       = any(current_time_onset>=play_bouts_table(:,1) & current_time_onset<=play_bouts_table(:,2),1);
        call_onset_regressor(hmm_n,:)                            = any(current_time_onset>=CallStats.BeginTimes & current_time_onset<=CallStats.EndTimes,1);
        self_onset_regressor(hmm_n,:)                            = any(current_time_onset>=slef_onset_offset(:,1) & current_time_onset<=slef_onset_offset(:,2),1);
        other_onset_regressor(hmm_n,:)                           = any(current_time_onset>=other_onset_offset(:,1) & current_time_onset<=other_onset_offset(:,2),1);
        self_speed_onset_regressor(hmm_n,allowed_index_freq)     = animal_speed_spect_time(entire_range_onset(allowed_index_freq));
        self_acc_onset_regressor(hmm_n,allowed_index_freq)       = animal_accel_spect_time(entire_range_onset(allowed_index_freq));
        other_speed_onset_regressor(hmm_n,allowed_index_freq)    = partner_speed_spect_time(entire_range_onset(allowed_index_freq));
        other_acc_onset_regressor(hmm_n,allowed_index_freq)      = partner_accel_spect_time(entire_range_onset(allowed_index_freq));
        animal_distance_onset_regressor(hmm_n,allowed_index_freq) = animal_dist_spect_time(entire_range_onset(allowed_index_freq));




        % --- Align to bout offset ---
        entire_range_offset                                     = round(loc_end+(hist_range(1)/spect_bin_size):loc_end+(hist_range(2)/spect_bin_size));
        if numel(entire_range_offset)>round(range(hist_range)/spect_bin_size)
            entire_range_offset(1) = [];
        end
        allowed_index_freq                                          = ismember(entire_range_offset,1:size(freq_pow,2));
        state_offset(hmm_n,allowed_index_freq)                      = freq_pow(entire_range_offset(allowed_index_freq));

        current_time_offset                                         = nan(1,size(call_offset_regressor,2));        
        current_time_offset(allowed_index_freq)                     = spect_time(entire_range_offset(allowed_index_freq));
        state_offset_regressor(hmm_n,:)                             = any(current_time_onset>=hmm_onset_offset(:,1) & current_time_onset<=hmm_onset_offset(:,2),1);       
        play_bout_offset_regressor(hmm_n,:)                         = any(current_time_offset>=play_bouts_table(:,1) & current_time_offset<=play_bouts_table(:,2),1);
        call_offset_regressor(hmm_n,:)                              = any(current_time_offset>=CallStats.BeginTimes & current_time_offset<=CallStats.EndTimes,1);
        self_offset_regressor(hmm_n,:)                              = any(current_time_offset>=slef_onset_offset(:,1) & current_time_offset<=slef_onset_offset(:,2),1);
        other_offset_regressor(hmm_n,:)                             = any(current_time_offset>=other_onset_offset(:,1) & current_time_offset<=other_onset_offset(:,2),1);
        self_speed_offset_regressor(hmm_n,allowed_index_freq)       = animal_speed_spect_time(entire_range_offset(allowed_index_freq));
        self_acc_offset_regressor(hmm_n,allowed_index_freq)         = animal_accel_spect_time(entire_range_offset(allowed_index_freq));
        other_speed_offset_regressor(hmm_n,allowed_index_freq)      = partner_speed_spect_time(entire_range_offset(allowed_index_freq));
        other_acc_offset_regressor(hmm_n,allowed_index_freq)        = partner_accel_spect_time(entire_range_offset(allowed_index_freq));       
        animal_distance_offset_regressor(hmm_n,allowed_index_freq)  = animal_dist_spect_time(entire_range_offset(allowed_index_freq));
    
    end
     

    %% -------------------- STORE RESULTS IN STRUCT --------------------
    if ch_n==1
        % First channel: initialize struct
        psth_struct = struct();
        for i = 1:numel(varNames2store)
            name = varNames2store{i};
            psth_struct.(name) = eval(name);
        end
       
    else
        % Additional channels: store as array of structs
       for i = 1:numel(varNames2store)
            name = varNames2store{i};
            psth_struct(ch_n).(name) = eval(name);
        end
     
    end
end
end
