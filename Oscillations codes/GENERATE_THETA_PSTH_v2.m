function psth_struct = GENERATE_THETA_PSTH_v2(current_dir)
% GENERATE_THETA_PSTH
% Computes theta-band (6–12 Hz) LFP power peri-event time histograms (PSTHs)
% around social play bouts and key behaviors. Outputs a struct with aligned data.

%% -------------------- SETUP AND PATHS --------------------
% Define behavior types of interest
play_behaviors = {'Pounce', 'CC','Boxing', 'Evasion','Pin', 'Escape', 'CB', 'CD'};

% Define data directories for synchronization, anatomical limits, and behavior
synch_directory  = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Synch data';
area_limit_table = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\Area_limits_GoodLooking.xlsx';
behavior_data    = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Behavior backups';

% Extract animal/session code information from current directory
animal_code        = strsplit(current_dir, '\');
animal_code        = animal_code{end};
animal_code_params = strsplit(animal_code, ' ');
animal_batch       = animal_code_params{1};
date               = animal_code_params{2};
repeated_animal    = animal_code_params{3};

%% -------------------- LOAD SYNCHRONIZATION MODEL --------------------
% Load mapping between video time and neural time
load([synch_directory,'\', animal_code, '\synch_model_video2NPX.mat'])

%% -------------------- LOAD BEHAVIOR DATA --------------------
% Load behavioral annotations
Behavior_file = [behavior_data,'\', animal_code,'.txt'];
Behavior = readtable(Behavior_file);

% Remove second column (irrelevant), rename columns
Behavior(:,2) = [];
Behavior.Properties.VariableNames = {'Animal', 'Start', 'End', 'Length', 'Type'};

% Setup bin size for behavior time binning
bin_size    = 0.01;
conv_length = 1;

% Merge related behaviors into consolidated categories
Behavior.Type2 = Behavior.Type;
Behavior.Type2(ismember(Behavior.Type2, {'Pounce_A', 'Pounce_B'}))   = {'Pounce'};
Behavior.Type2(ismember(Behavior.Type2, {'Pounce_Ai', 'Pounce_Bi'})) = {'PounceI'};
Behavior.Type2(ismember(Behavior.Type2,''))                          = {'Other'};

% Remove irrelevant entries
Behavior(ismember(Behavior.Animal, 'Reversal'),:) = [];

% Get unique animal types (ignore session structure)
animal_types = unique(Behavior.Animal);
animal_types(ismember(animal_types,'Session_structure')) = [];

% Convert behavior timestamps to neural time using sync model
Behavior.Start = predict(synch_model_video2NPX, Behavior.Start);
Behavior.End   = predict(synch_model_video2NPX, Behavior.End);

% Identify partner animals (excluding repeated_animal)
partner_names = animal_types;
partner_names(ismember(animal_types, repeated_animal)) = [];

% Store parameters in config struct for later use
config.Behavior        = Behavior;
config.repeated_animal = repeated_animal;
config.animal_types    = animal_types;
config.play_behaviors  = play_behaviors;
config.beh_bin         = bin_size;
config.conv_length     = conv_length;
config.behavior_window = 0;

% Identify play bouts from behavior data
[play_bouts_table] = play_bout(config);

%% -------------------- LOAD LFP DATA --------------------
disp('LOADING LFP')
if exist([current_dir,'\','LFP_PAG.mat'], 'file')==2
    % Preprocessed LFP file exists
    NPX_Type = 2;
    load([current_dir,'\','LFP_PAG.mat'], 'LFP')
elseif exist([current_dir,'\','LFP_PAG.dat'], 'file')==2
    % Load raw binary LFP file
    NPX_Type = 1;
    file_pointer = fopen([current_dir,'\','LFP_PAG.dat'], 'r');
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
    load([current_dir,'\ChannelMap.mat'], 'xcoords', 'ycoords','chanMap')
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
% Spectrogram parameters
hist_range      = [-20 20];       % peri-event window for onset/offset (s)
range_time_wrap = [-5 5];         % time-wrap range (s)
n_bins_time_wrap = range(range_time_wrap)*100;

% Extract LFP from mid-PAG channel(s)
PAG_LFP = double(LFP(mid_PAG_channel,:));
clear LFP
LFP_time = (1:size(PAG_LFP,2))/2500; % sample rate: 2.5 kHz

% Spectrogram parameters
wind_length     = .250;    % 250 ms
wind_overlap    = 0.240;   % 240 ms overlap
spect_bin_size  = wind_length-wind_overlap;
min_separation  = .200;    % minimal separation between behaviors

f          = 4:.1:15;      % frequency range for spectrogram
freq_range = [6 12];       % theta band

% Identify theta indices in spectrogram
f_index = f>=freq_range(1) & f<=freq_range(2);

% Only compute spectrogram over LFP range covering all play bouts
range2exctract = LFP_time>=min(play_bouts_table(:))+hist_range(1) & LFP_time<=max(play_bouts_table(:))+hist_range(2);

% Initialize output struct
psth_struct = [];

for ch_n=1:numel(mid_PAG_channel)
    disp('Estimating spectrogram')
    % Compute spectrogram for selected LFP channel
    [pow_spectrogram,~,spect_time]  = spectrogram(PAG_LFP(ch_n,range2exctract),wind_length*2500, wind_overlap*2500, f,2500);
    spect_time = spect_time+min(play_bouts_table(:))+hist_range(1);
    pow_spectrogram = abs(pow_spectrogram);

    % Extract theta-band power (log-scaled, smoothed)
    theta_pow = mean(log10(pow_spectrogram(f_index,:)));
    theta_pow = movmean(theta_pow,1/max(freq_range));

    %% -------------------- COMPUTE PERI-EVENT THETA PSTHs --------------------
    % Allocate arrays for different peri-event alignments
    play_bout_onset       = nan(size(play_bouts_table,1), round(range(hist_range)/spect_bin_size));
    animal_behavior_onset = nan(size(play_bouts_table,1), round(range(hist_range)/spect_bin_size));
    play_bout_offset      = nan(size(play_bouts_table,1), round(range(hist_range)/spect_bin_size));
    play_bout_tw_this     = nan(size(play_bouts_table,1), round(range(range_time_wrap)/spect_bin_size) + n_bins_time_wrap);
    three_point_tw        = nan(size(play_bouts_table,1), round(range(range_time_wrap)/spect_bin_size) + 2*n_bins_time_wrap);
    first_animal_behavior = nan(size(play_bouts_table,1),2);

    % Loop over play bouts
    for pb_n=1:size(play_bouts_table,1)
        % Get bout start/end
        play_bout_start = play_bouts_table(pb_n,1);
        play_bout_end   = play_bouts_table(pb_n,2);
        [~,loc_start] = min(abs(spect_time-play_bout_start));
        [~,loc_end]   = min(abs(spect_time-play_bout_end));

        % --- Align to bout onset ---
        entire_range = round(loc_start+(hist_range(1)/spect_bin_size):loc_start+(hist_range(2)/spect_bin_size));
        allowed_index_theta = ismember(entire_range,1:size(theta_pow,2));
        play_bout_onset(pb_n,allowed_index_theta) = theta_pow(entire_range(allowed_index_theta));

        % --- Align to bout offset ---
        entire_range = round(loc_end+(hist_range(1)/spect_bin_size):loc_end+(hist_range(2)/spect_bin_size));
        allowed_index_theta = ismember(entire_range,1:size(theta_pow,2));
        play_bout_offset(pb_n,allowed_index_theta) = theta_pow(entire_range(allowed_index_theta));

        % --- Time-warp entire bout ---
        pre_time  = loc_start+ round(range_time_wrap(1)/spect_bin_size):loc_start-1;
        post_time = loc_end+1:loc_end+round(range_time_wrap(2)/spect_bin_size);
        in_between_time = loc_start:loc_end;
        time_wrapped_theta = interp1(in_between_time, theta_pow(in_between_time), linspace(loc_start,loc_end,n_bins_time_wrap));
        play_bout_tw_this(pb_n,:) =  [theta_pow(pre_time),time_wrapped_theta, theta_pow(post_time)];

        % --- Find first behavior within bout ---
        beh1 = min(find(Behavior.Start>play_bout_start & ismember(Behavior.Animal, repeated_animal)));
        beh2 = min(find(Behavior.Start>play_bout_start & ismember(Behavior.Animal, repeated_animal) & ismember(Behavior.Type2, play_behaviors)));
        beh_start = Behavior.Start(beh1);

        if ~isempty(beh1) && beh_start<play_bout_end
            first_animal_behavior(pb_n,1) = beh1;

            % Align to first behavior onset
            [~,loc_beh_start] =  min(abs(spect_time-beh_start));
            entire_range = round(loc_beh_start+(hist_range(1)/spect_bin_size):loc_beh_start+(hist_range(2)/spect_bin_size));
            allowed_index_theta = ismember(entire_range,1:size(theta_pow,2));
            animal_behavior_onset(pb_n,allowed_index_theta) = theta_pow(entire_range(allowed_index_theta));

            % --- If behavior occurs mid-bout: split bout into two segments ---
            if beh_start>play_bout_start+min_separation && beh_start<play_bout_end
                pre_time  = loc_start+ round(range_time_wrap(1)/spect_bin_size):loc_start-1;
                post_time = loc_end+1:loc_end+round(range_time_wrap(2)/spect_bin_size);

                in_between_time1 = loc_start:loc_beh_start-1;
                time_wrapped_theta1 = interp1(in_between_time1, theta_pow(in_between_time1), linspace(loc_start,loc_beh_start-1,n_bins_time_wrap));

                in_between_time2 = loc_beh_start:loc_end;
                time_wrapped_theta2 = interp1(in_between_time2, theta_pow(in_between_time2), linspace(loc_beh_start,loc_end,n_bins_time_wrap));

                three_point_tw(pb_n,:) = [theta_pow(pre_time),time_wrapped_theta1,time_wrapped_theta2, theta_pow(post_time)];
            end
        end
        if ~isempty(beh2)
            first_animal_behavior(pb_n,2) = beh2;
        end 
    end
    
    %% -------------------- STORE RESULTS IN STRUCT --------------------
    if ch_n==1
        % First channel: initialize struct
        psth_struct.play_bout_onset         = play_bout_onset;
        psth_struct.play_bout_offset        = play_bout_offset;
        psth_struct.play_bout_tw_this       = play_bout_tw_this;
        psth_struct.hist_range              = hist_range;
        psth_struct.range_time_wrap         = range_time_wrap;
        psth_struct.n_bins_time_wrap        = n_bins_time_wrap;
        psth_struct.wind_length             = wind_length;
        psth_struct.wind_overlap            = wind_overlap;
        psth_struct.ch                      = mid_PAG_channel(ch_n);
        psth_struct.play_bouts_table        = play_bouts_table;
        psth_struct.Behavior                = Behavior;
        psth_struct.animal_behavior_onset   = animal_behavior_onset;
        psth_struct.first_animal_behavior   = first_animal_behavior;
        psth_struct.f                       = f;
        psth_struct.freq_range              = freq_range;
        psth_struct.min_separation          = min_separation;
        psth_struct.three_point_tw          = three_point_tw;
    else
        % Additional channels: store as array of structs
        psth_struct(ch_n).play_bout_onset           = play_bout_onset;
        psth_struct(ch_n).play_bout_offset          = play_bout_offset;
        psth_struct(ch_n).play_bout_tw_this         = play_bout_tw_this;
        psth_struct(ch_n).hist_range                = hist_range;
        psth_struct(ch_n).range_time_wrap           = range_time_wrap;
        psth_struct(ch_n).n_bins_time_wrap          = n_bins_time_wrap;
        psth_struct(ch_n).wind_length               = wind_length;
        psth_struct(ch_n).wind_overlap              = wind_overlap;
        psth_struct(ch_n).ch                        = mid_PAG_channel(ch_n);
        psth_struct(ch_n).play_bouts_table          = play_bouts_table;
        psth_struct(ch_n).Behavior                  = Behavior;
        psth_struct(ch_n).animal_behavior_onset     = animal_behavior_onset;
        psth_struct(ch_n).first_animal_behavior     = first_animal_behavior;
        psth_struct(ch_n).f                         = f;
        psth_struct(ch_n).freq_range                = freq_range;
        psth_struct(ch_n).min_separation            = min_separation;
        psth_struct(ch_n).three_point_tw            = three_point_tw;
    end
end
end
