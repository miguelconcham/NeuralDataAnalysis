function psth_struct = GENERATE_LOCKED_PSTH(current_dir,Hd,hist_range)


%% params
% hist_range      = [-5 5];
fs              = 2500;
play_behaviors  = {'Pounce', 'CC','Boxing', 'Evasion','Pin', 'Escape', 'CB', 'CD'};

%%

synch_directory     = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Synch data';
area_limit_table    = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\Area_limits_GoodLooking.xlsx';
behavior_data       = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Behavior backups';
% npx_raw_data = 
animal_code         = strsplit(current_dir, '\');
animal_code         = animal_code{end};
animal_code_params  = strsplit(animal_code, ' ');
animal_batch        = animal_code_params{1};
date                = animal_code_params{2};
repeated_animal     = animal_code_params{3};

%% load synch from synch folder
load([synch_directory,'\', animal_code, '\synch_model_video2NPX.mat'])

%% 2 Load bheavior data and estimate play bouts


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

animal_types            = unique(Behavior.Animal);

animal_types(ismember(animal_types,'Session_structure'))                =[];

Behavior.Start          = predict(synch_model_video2NPX, Behavior.Start);
Behavior.End            = predict(synch_model_video2NPX, Behavior.End);

partner_names           = animal_types;
partner_names(ismember(animal_types, repeated_animal))                  = [];

config.Behavior         = Behavior;
config.repeated_animal  = repeated_animal;
config.animal_types     = animal_types        ;
config.play_behaviors   = play_behaviors      ;
config.beh_bin          = bin_size             ;
config.conv_length      = conv_length;
config.behavior_window  = 0;


[play_bouts_table]      = play_bout(config);




%%  load lfp from current dir
disp('LOADING LFP')
if exist([current_dir,'\','LFP_PAG.mat'], 'file')==2

    NPX_Type        = 2;
    load([current_dir,'\','LFP_PAG.mat'], 'LFP')
elseif exist([current_dir,'\','LFP_PAG.dat'], 'file')==2
    NPX_Type        = 1;
    file_pointer    = fopen([current_dir,'\','LFP_PAG.dat'], 'r');
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
    load([current_dir,'\ChannelMap.mat'], 'xcoords', 'ycoords','chanMap')
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


PAG_LFP         = double(LFP(mid_PAG_channel,:));
clear LFP
LFP_time        = (1:size(PAG_LFP,2))/2500;
psth_struct     = [];
for ch_n=1:numel(mid_PAG_channel)
    disp('Filtering Signal')
    filtered_signal = filtfilt(Hd.Coefficients, 1, PAG_LFP(ch_n,:));
    filtered_signal = hilbert(filtered_signal);



    phase_data1         = angle(filtered_signal);
    amplitud_data1      = abs(filtered_signal);

    std_amp1 = std(amplitud_data1);
    [~,max_locs] = findpeaks(amplitud_data1, 'MinPeakProminence',.5*std_amp1, 'MinPeakDistance', fs/(Hd.CutoffFrequency2  )) ;
    original_distribution   = phase_data1(max_locs);       
    mean_angle_original     = angle(mean(exp(1i*original_distribution)));
    mean_angle_original     = mod( mean_angle_original+2*pi,2*pi);  
    phase_data1             = mod(phase_data1  - mean_angle_original + 5*pi , 2*pi) - pi; %% centering step

    filtered_signal = amplitud_data1.*(exp(1i*phase_data1));
    disp('Estiamting PSTH')
   
   
  

 


    play_bout_onset          = nan(size(play_bouts_table,1), round(range(hist_range)*fs));
    play_bout_offset         = nan(size(play_bouts_table,1), round(range(hist_range)*fs));
    play_bout_onset_raw      = nan(size(play_bouts_table,1), round(range(hist_range)*fs));
    play_bout_offset_raw     = nan(size(play_bouts_table,1), round(range(hist_range)*fs));



    for pb_n=1:size(play_bouts_table,1)

        play_bout_start     = play_bouts_table(pb_n,1);
        play_bout_end       = play_bouts_table(pb_n,2);
        [~,loc_start]       = min(abs(LFP_time-play_bout_start));
        [~,loc_end]         = min(abs(LFP_time-play_bout_end));

        entire_range        = round(loc_start+(hist_range(1)*fs)+1:loc_start+(hist_range(2)*fs));
        allowed_index_fres = ismember(entire_range,1:size(filtered_signal,2));
        play_bout_onset(pb_n,allowed_index_fres)        = filtered_signal(entire_range(allowed_index_fres));
        play_bout_onset_raw(pb_n,allowed_index_fres)    =  PAG_LFP(ch_n,entire_range(allowed_index_fres));
       


        entire_range        = round(loc_end+(hist_range(1)*fs)+1:loc_end+(hist_range(2)*fs));
        allowed_index_fres = ismember(entire_range,1:size(filtered_signal,2));
        play_bout_offset(pb_n,allowed_index_fres) = filtered_signal(entire_range(allowed_index_fres));
        play_bout_offset_raw(pb_n,allowed_index_fres) =  PAG_LFP(ch_n,entire_range(allowed_index_fres));
     
    end
    
    if ch_n==1
        psth_struct.play_bout_onset         = play_bout_onset;
        psth_struct.play_bout_offset        = play_bout_offset; 
        psth_struct.play_bout_onset_raw     = play_bout_onset_raw;
        psth_struct.play_bout_offset_raw    = play_bout_offset_raw;
        psth_struct.hist_range              = hist_range;       
        psth_struct.ch                      = mid_PAG_channel(ch_n);
        psth_struct.play_bouts_table        = play_bouts_table;
        psth_struct.Behavior                = Behavior;
  


    else
        psth_struct(ch_n).play_bout_onset         = play_bout_onset;
        psth_struct(ch_n).play_bout_offset        = play_bout_offset; 
        psth_struct(ch_n).play_bout_onset_raw     = play_bout_onset_raw;
        psth_struct(ch_n).play_bout_offset_raw    = play_bout_offset_raw;
        psth_struct(ch_n).hist_range              = hist_range;       
        psth_struct(ch_n).ch                      = mid_PAG_channel(ch_n);
        psth_struct(ch_n).play_bouts_table        = play_bouts_table;
        psth_struct(ch_n).Behavior                = Behavior;


    end

end
end
