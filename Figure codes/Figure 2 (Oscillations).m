figure_dir          = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Codes\Figure codes\Figure 2 Inputs';
synch_directory     = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Synch data';
hmm_raw_data        = ['\\experimentfs.bccn-berlin.pri\experiment\PlayN' ...
    'euralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\HMM data\HMM raw data'];
call_folder         = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\CallDetectionBackup';
behavior_folder     = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Behavior backups';
npx_folder          = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\NPX data\NPX raw data';
area_limit_table    = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\Area_limits_GoodLooking.xlsx';
saving_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\Theta psth';

animal_code     = 'B1S3 1008 Single';
animal_code_params = strsplit(animal_code, ' ');
animal_batch       = animal_code_params{1};
date               = animal_code_params{2};
repeated_animal    = animal_code_params{3};


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
%% -------------------- LOAD SYNCHRONIZATION MODEL --------------------
% Load mapping between video time and neural time
load([synch_directory,'\', animal_code, '\synch_model_video2NPX.mat'])
load([synch_directory,'\', animal_code, '\synch_model_audio2NPX.mat'])

%% estimae raw lfp data

low_pass_freq = 100;


start_event = 337-4; %in audio time
end_event   =338.5 +1; %in audio time
start_event  =predict(synch_model_audio2NPX,start_event);
end_event       = predict(synch_model_audio2NPX,end_event);

sr = 2500;
start_event_index = round(start_event*sr);
end_event_index     = round(end_event*sr);
slected_lfp = LFP(mid_PAG_channel,start_event_index:end_event_index);
slected_lfp = movmean(slected_lfp,sr/low_pass_freq );
time = (start_event_index:end_event_index)/sr;
theta_band = [6 12];
delta_band = [1 5];

figure
subplot(2,1,1)
plot(time,slected_lfp,'k')
filter_order = 2000;
Hd_theta = designfilt('bandpassfir', ...
'FilterOrder', filter_order, ...
'CutoffFrequency1', theta_band(1), ...
'CutoffFrequency2', theta_band(2), ...
'SampleRate', sr, ...
'DesignMethod', 'window', ...
'Window', 'hamming');

Hd_delta = designfilt('bandpassfir', ...
'FilterOrder', filter_order, ...
'CutoffFrequency1', delta_band(1), ...
'CutoffFrequency2', delta_band(2), ...
'SampleRate', sr, ...
'DesignMethod', 'window', ...
'Window', 'hamming');

filtered_signal_theta = filtfilt(Hd_theta.Coefficients, 1, LFP(mid_PAG_channel,start_event_index:end_event_index));
filtered_signal_delta = filtfilt(Hd_delta.Coefficients, 1, LFP(mid_PAG_channel,start_event_index:end_event_index));

hold on
plot(time,filtered_signal_theta,'b')
plot(time,filtered_signal_delta,'r')

%% estimate pow spectr with confident intervals

freqRange = [.1 100];   % Desired frequency range (Hz)
winLength_sec = 10;
% Compute power spectrum
timeBW = 3;
[f,meanPxx,ci] = avg_spectrum_CI_pmtm(LFP(mid_PAG_channel,:),sr,freqRange,round(winLength_sec*sr),0,1000,timeBW);


%% plot pow spectrum


subplot(2,1,2)
plot(f,10*log10(meanPxx)); hold on
plot(f,10*log10(ci(:,1)),'--r',f,10*log10(ci(:,2)),'--r')
xlabel('Frequency (Hz)'); ylabel('Power (dB)');
xlim([0 20])

%% save pow for figure


print(gcf,'-vector','-dsvg',[figure_dir,'\thet delta scillations raw.svg'])

%% Speed theta relation (structre obtained using function "estiamte all speed theta relation"

load([saving_folder,'\psth_structure_speed_theta.mat'],'psth_structure')
load([saving_folder,'\animal_names_speed_theta.mat'],'animal_names')

%% obtain theta increas eduirng play (mixed effect linear model)

stasts_table = cell(numel(psth_structure),6);

for j=1:numel(psth_structure)

    stasts_table(j,:) = { psth_structure(j).lm.Coefficients.Estimate(4), ...
        psth_structure(j).lm.Coefficients.pValue(4),...
        psth_structure(j).lm.Coefficients.tStat(4),...
        animal_names{j,:}};

end

stasts_table =cell2table(stasts_table);
stasts_table.Properties.VariableNames = {'Estimate','pValue','tStat','Animal','Partner','Electrode'};

stasts_table = stasts_table(stasts_table.Electrode==1,:);

lme = fitlme(stasts_table, 'Estimate ~ 1 + (1|Animal)');
coef = fixedEffects(lme);       % estimated mean
ci_estimate = coefCI(lme);               % confidence interval
pval = lme.Coefficients.pValue; % p-value for intercept

fprintf('Mean = %.3f, 95%% CI [%.3f, %.3f], p = %.4f\n', coef, ci(1), ci(2), pval);



%% Compute speed and theta  
n_grid = 101;
distance_grid = linspace(-10, 10, n_grid);

data_together = [];
mean_powers_play    = nan(numel(psth_structure) ,n_grid);
mean_powers_noplay  = nan(numel(psth_structure) ,n_grid);
for j=1:numel(psth_structure)


    tbl = psth_structure(j).model_data;
    tbl = tbl(~isnan(tbl.Speed),:);

    Y = tbl.Power;
    Y = (Y - mean(Y, 'omitmissing'))/std(Y, 'omitmissing');
    X = tbl.Speed;
    Xa = X(tbl.Play=='true');
    Ya = Y(tbl.Play=='true');


    Xb = X(tbl.Play=='false');
    Yb =  Y(tbl.Play=='false');


    data_together = [data_together;[X Y]];

    edges = [distance_grid Inf];

    % Assign each sample to a bin
    [~,~,binA] = histcounts(Xa, edges);
    [~,~,binB] = histcounts(Xb, edges);

    % Pre-allocate as NaN
    meanA = nan(size(distance_grid));
    meanB = nan(size(distance_grid));

    % Compute mean per bin (only for bins with data)
    for i = 1:numel(distance_grid)
        meanA(i) = mean(Ya(binA == i), 'omitnan');
        meanB(i) = mean(Yb(binB == i), 'omitnan');
    end


    mean_powers_play(j,:)    = meanA;
    mean_powers_noplay(j,:)  = meanB;
end
%% now plot
[bin_count, ~, samples2plot] = histcounts(data_together(:,1), -5:.1:10);

figure
bins_with_values = unique(samples2plot)'
max_count = 100;
min_count = 20;

indexes2plot = [];

for j=bins_with_values
    indexes= find(samples2plot==j);
    if numel(indexes)>=max_count

        indexes  = datasample(indexes,max_count );
    end
    if numel(indexes)>=min_count
        indexes2plot = [indexes2plot;indexes];
    end
end




plot(data_together(indexes2plot,1), data_together(indexes2plot,2), '.k', 'MarkerSize',    .1)
axis xy
%% save figure
print(gcf,'-vector','-dsvg',[figure_dir,'\speed and theta.svg'])

%% plot thet during play vs theta during no play for fixed speed bins
figure
subplot(1,2,1)
plot(distance_grid, mean_powers_play-mean_powers_noplay, ':k');
hold on
plot(distance_grid, mean(mean_powers_play-mean_powers_noplay, 'omitmissing'), 'k');

[h, p, ci_speed_theta] = ttest(mean_powers_play-mean_powers_noplay);

hold on
no_nan = ~any(isnan(ci_speed_theta));
fill([distance_grid(no_nan) fliplr(distance_grid(no_nan))], [ci_speed_theta(1,no_nan) fliplr(ci_speed_theta(2,no_nan))], 'k','FaceAlpha',.25, 'EdgeColor','none')
ylim([-1 1])
plot(distance_grid, distance_grid*0,'k' )
xlim([-1 4])
subplot(1,2,2)
rand_x = rand(size(stasts_table.Estimate));
plot(rand_x, stasts_table.Estimate, 'k.')

xlim([-.75 1.75])
hold on
errorbar(.5, mean(stasts_table.Estimate), ci_estimate(1),ci_estimate(2), 'r')
plot([.25 .75],[1 1]*mean(stasts_table.Estimate), 'r')
ylim([-.1 .1])
title(pval)

%% save figure
print(gcf,'-vector','-dsvg',[figure_dir,'\theta beyond speed.svg'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% THETA SECTION NOW %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load psth of theta lock to play bout  (function: Estimamte_psth_all_files)
disp('Loading')
load([saving_folder,'\psth_structure.mat'],'psth_structure');
load([saving_folder,'\animal_names.mat'],'animal_names');
disp('Loading Ready')

%% merging_psth
smooth_wind = 20;
baseline_range = [-2 0]
animal_label = {'B1D1','B1S3','B2S2','B3D2', 'B4S2', 'B4D4'};
electorde_numner = [1 2];
bin_size = psth_structure(1).wind_length - psth_structure(1).wind_overlap;
psth_ranges = psth_structure(1).hist_range;
wrap_range = psth_structure(1).range_time_wrap;
time = psth_ranges(1):bin_size:psth_ranges(2)+bin_size;
baseline_index = time<baseline_range(2) & time>baseline_range(1);
baseline_index_time_wrap = 1:round((abs(wrap_range(1))/bin_size));
all_psth_onset          = [];
all_psth_onset_behavior = [];
all_psth_onset_only_playobut    = [];

all_psth_offset         = [];
all_psth_tw             = [];
all_psth_tw_3points     = [];
all_play_bouts          = [];
time_wrap_time          = [(baseline_index_time_wrap*bin_size) + wrap_range(1),linspace(0,1,psth_structure(1).n_bins_time_wrap),1 + (1:round((abs(wrap_range(2))/bin_size)))*bin_size];
time_wrap_3_points      = [(baseline_index_time_wrap*bin_size) + wrap_range(1),linspace(0,1-1/psth_structure(1).n_bins_time_wrap,psth_structure(1).n_bins_time_wrap), ...
    linspace(1,2-1/psth_structure(1).n_bins_time_wrap,psth_structure(1).n_bins_time_wrap),2 + (1:round((abs(wrap_range(2))/bin_size)))*bin_size];


animal_index = [];
electrode_index = []
for j=1:numel(psth_structure)

    if contains(animal_names{j},animal_label)

        animal_num      = find(cell2mat(cellfun(@(x) contains(animal_names{j},x), animal_label, 'UniformOutput',false)));
        electrode_num   = animal_names{j,2}  ;
        this_animal_playbouts = psth_structure(j).play_bouts_table;
        this_animal_lengths = diff(this_animal_playbouts');

        this_psth_onset         = psth_structure(j).play_bout_onset;
        this_psth_onset_onlypb  = this_psth_onset;

        animal_index = [animal_index;repmat(animal_num,size(this_psth_onset,1),1)];
        electrode_index = [electrode_index;repmat(electrode_num,size(this_psth_onset,1),1)];

        for trial=1:size(this_psth_onset,1)
            this_psth_onset(trial,:) = ( this_psth_onset(trial,:) - mean( this_psth_onset(trial,baseline_index)))/std( this_psth_onset(trial,baseline_index));
            this_psth_onset(trial,:) = movmean(this_psth_onset(trial,:), smooth_wind);
            this_psth_onset_onlypb(trial,:) = this_psth_onset(trial,:);
            this_psth_onset_onlypb(trial,time> this_animal_lengths(trial)) = NaN;
        end
        all_psth_onset      = [all_psth_onset; this_psth_onset];
        all_psth_onset_only_playobut = [all_psth_onset_only_playobut; this_psth_onset_onlypb];

        this_psth_onset     = psth_structure(j).play_bout_onset;
        this_psth_offset    = psth_structure(j).play_bout_offset;
        for trial=1:size(this_psth_offset,1)
            this_psth_offset(trial,:) = ( this_psth_offset(trial,:) - mean( this_psth_onset(trial,baseline_index)))/std( this_psth_onset(trial,baseline_index));
            this_psth_offset(trial,:) = movmean(this_psth_offset(trial,:), smooth_wind);
        end
        all_psth_offset = [all_psth_offset; this_psth_offset];

        this_psth_tw = psth_structure(j).play_bout_tw_this;
        for trial=1:size(this_psth_tw,1)
            this_psth_tw(trial,:) = ( this_psth_tw(trial,:) - mean( this_psth_tw(trial,baseline_index_time_wrap)))/std( this_psth_tw(trial,baseline_index_time_wrap));
        end
        all_psth_tw = [all_psth_tw; this_psth_tw];


        this_psth_tw = psth_structure(j).three_point_tw;
        for trial=1:size(this_psth_tw,1)
            this_psth_tw(trial,:) = ( this_psth_tw(trial,:) - mean( this_psth_tw(trial,baseline_index_time_wrap)))/std( this_psth_tw(trial,baseline_index_time_wrap));
        end
        all_psth_tw_3points = [all_psth_tw_3points; this_psth_tw];



        this_psth_ab = psth_structure(j).animal_behavior_onset;
        for trial=1:size(this_psth_ab,1)
            this_psth_ab(trial,:) = ( this_psth_ab(trial,:) - mean( this_psth_ab(trial,baseline_index_time_wrap)))/std( this_psth_ab(trial,baseline_index_time_wrap));
            this_psth_ab(trial,:) = movmean(this_psth_ab(trial,:), smooth_wind);
        end
        all_psth_onset_behavior = [all_psth_onset_behavior; this_psth_ab];

        all_play_bouts = [all_play_bouts;this_animal_playbouts];
    end
end

play_bout_length = diff(all_play_bouts')';


[sorted_play_bout_length, order] = sort(play_bout_length);

%% plot single animals  (and obtain mean response per animal)
X_lim = [-2 2]
figure
min_length = .0;
stacked_mean_onset = [];
% what2plot = all_psth_onset_behavior; %Select what to plot
what2plot = all_psth_onset_only_playobut;
for an= 1:numel(animal_label)
    animal_bool = animal_index==an;
    length_bool = play_bout_length>min_length;
    electrode_bool = electrode_index==1;
    [sorted_play_bout_length, order] = sort(play_bout_length(animal_bool & length_bool & electrode_bool,:));
    subplot(5,numel(animal_label),(1:numel(animal_label):2*numel(animal_label)) + an-1)

    array = what2plot(animal_bool & length_bool & electrode_bool,:);
    imagesc(time,1:numel(sorted_play_bout_length),array(order,:) )
    xlim(X_lim)
    clim([-2 2])
    axis xy
    hold on
    plot([0 0],[1 numel(sorted_play_bout_length)], 'w')
    plot(sorted_play_bout_length,1:numel(sorted_play_bout_length), 'w')
    title(animal_label{an})

    subplot(5,numel(animal_label),((2*numel(animal_label) + 1):numel(animal_label):5*numel(animal_label)) + an-1)

    [~, ~, ci]  = ttest(array);
    no_nan = ~any(isnan(ci));
    fill([time(no_nan) fliplr(time(no_nan))], [ci(1,no_nan) fliplr(ci(2,no_nan))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time,mean(array, 'omitmissing'), 'k')
    xlim(X_lim)
    stacked_mean_onset = [stacked_mean_onset;mean(array, 'omitmissing')];
end

%% load mixed model effect

% load([saving_folder,'\results_play_bout.mat'],'results'); %using all time points
load([saving_folder,'\results_play_bout_onlybout.mat'],'results'); %using only time points within playbouts
limited_time = results.time;
est = results.est;
ci = results.ci;
pvals_fdr = results.pvals;
%% plot all together

X_lim = [-1 3];
alpha = 0.05;
figure
min_length = .0;
fill_lim = [.35 .4];

length_bool = play_bout_length>min_length;
electrode_bool = electrode_index==1;
[sorted_play_bout_length, order] = sort(play_bout_length( length_bool & electrode_bool,:));

subplot(2,1,1)
array = what2plot( length_bool & electrode_bool,:);
imagesc(time,1:numel(sorted_play_bout_length),array(order,:) )

xlim(X_lim)
clim([-1 2])
axis xy
hold on
plot([0 0],[1 numel(sorted_play_bout_length)], 'w')
plot(sorted_play_bout_length,1:numel(sorted_play_bout_length), 'w')
title('All data')

subplot(2,1,2)

[~, ~, ci]  = ttest(array);
fill([time fliplr(time)], [ci(1,:) fliplr(ci(2,:))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
hold on
plot(time,mean(array, 'omitmissing'), 'k')
xlim(X_lim)

plot(time,stacked_mean_onset, 'k:')
hold on
plot(time,mean(stacked_mean_onset), 'k:')
xlim(X_lim)

sig_idx = pvals_fdr < alpha;
limited_time = results.time;

borders = [ 0 sig_idx' 0];
beg_index = find(diff(borders)==1);

end_index = find(diff(borders)==-1)-1;

for bn = 1:numel(beg_index)
    time2plot = limited_time(beg_index(bn):end_index(bn))';
    fill([time2plot fliplr(time2plot)], [time2plot*0+fill_lim(1) time2plot*0+fill_lim(2)], 'r', 'EdgeColor','none')
end

hold on
y_lim =ylim;
plot([0 0],y_lim, 'k' )
ylim tight

%% save figure
print(gcf,'-vector','-dsvg',[figure_dir,'\play and theta only playoubt.svg'])

%% now load call data (from "Estimate psth call all animals")
load([saving_folder,'\psth_structure_call.mat'],'psth_structure');
load([saving_folder,'\animal_names_call.mat'],'animal_names');

%% mergin data

smooth_wind             = 20;
baseline_range          = [-2 0]
animal_label            = {'B1D1','B1S3','B2S2','B3D2', 'B4S2'};
electorde_number        = [1 2];
bin_size                = psth_structure(1).wind_length - psth_structure(1).wind_overlap;
psth_ranges             = psth_structure(1).hist_range;
time                    = psth_ranges(1):bin_size:psth_ranges(2)+bin_size;
baseline_index          = time<baseline_range(2) & time>baseline_range(1);
all_psth_onset          = [];
all_psth_offset         = [];
all_onset_regressors    = [];
all_offset_regressors   = [];
all_Calls               = [];

electrode_index  = [];
animal_index = [];
for j=1:numel(psth_structure)

    if contains(animal_names{j},animal_label)

        animal_num      = find(cell2mat(cellfun(@(x) contains(animal_names{j},x), animal_label, 'UniformOutput',false)));
        electrode_num   = animal_names{j,2};

        this_psth_onset         = psth_structure(j).call_onset;
        animal_index = [animal_index;repmat(animal_num,size(this_psth_onset,1),1)];
        electrode_index = [electrode_index;ones(size(this_psth_onset,1),1)*electrode_num];
        for trial=1:size(this_psth_onset,1)
            this_psth_onset(trial,:) = ( this_psth_onset(trial,:) - mean( this_psth_onset(trial,baseline_index)))/std( this_psth_onset(trial,baseline_index));
            this_psth_onset(trial,:) = movmean(this_psth_onset(trial,:), smooth_wind);
        end
        all_psth_onset      = [all_psth_onset; this_psth_onset];

        this_psth_onset     = psth_structure(j).call_onset;
        this_psth_offset    = psth_structure(j).call_offset;
        for trial=1:size(this_psth_offset,1)
            this_psth_offset(trial,:) = ( this_psth_offset(trial,:) - mean( this_psth_onset(trial,baseline_index)))/std( this_psth_onset(trial,baseline_index));
            this_psth_offset(trial,:) = movmean(this_psth_offset(trial,:), smooth_wind);
        end
        all_psth_offset = [all_psth_offset; this_psth_offset];


        all_onset_regressors = [all_onset_regressors; psth_structure(j).call_onset_regressor];
        all_offset_regressors = [all_offset_regressors; psth_structure(j).call_onset_regressor];


        all_Calls = [all_Calls;psth_structure(j).CallStats];
    end
end

call_lengths = all_Calls.CallLengths;


[sorted_call_lengths, order] = sort(call_lengths);

disp('merge done')
%% ploting each animal (and obtain mean response per animal)

stacked_mean = [];
X_lim = [-2 .5];
figure
min_length = .0;
for an= 1:numel(animal_label)
    animal_bool = animal_index==an;
    length_bool = call_lengths>min_length;
    electrode_bool = electrode_index==1;
    [sorted_call_lengths, order] = sort(call_lengths(animal_bool & length_bool & electrode_bool,:));
    subplot(5,numel(animal_label),(1:numel(animal_label):2*numel(animal_label)) + an-1)

    array = all_psth_onset(animal_bool & length_bool & electrode_bool,:);
    imagesc(time,1:numel(sorted_call_lengths),array(order,:) )
    xlim(X_lim)
    clim([-2 2])
    axis xy
    hold on
    plot([0 0],[1 numel(sorted_call_lengths)], 'w')
    plot(sorted_call_lengths,1:numel(sorted_call_lengths), 'w')
    title(animal_label{an})

    subplot(5,numel(animal_label),((2*numel(animal_label) + 1):numel(animal_label):5*numel(animal_label)) + an-1)

    [~, ~, ci]  = ttest(array);
    no_nan = ~any(isnan(ci));
    fill([time(no_nan) fliplr(time(no_nan))], [ci(1,no_nan) fliplr(ci(2,no_nan))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time,mean(array,'omitmissing'), 'k')
    yyaxis right
    plot(time, mean(all_onset_regressors(animal_bool & length_bool,:)), 'r')
    xlim(X_lim)

    stacked_mean = [stacked_mean;mean(array,'omitmissing')];
end

%% load result file for ploting all together

load([saving_folder,'\results_call.mat'],'results');

limited_time = results.time;
est = results.est;
ci = results.ci;
pvals_fdr = results.pvals;
%% now plot all together
figure
fill_lim = [.15 .18]
X_lim = [-1 .5];
min_length = 0;
animal_bool = animal_index==an;
length_bool = call_lengths>min_length;
electrode_bool = electrode_index==1;
[sorted_call_lengths, order] = sort(call_lengths( length_bool & electrode_bool,:));
subplot(2,1,1)

array = all_psth_onset( length_bool & electrode_bool,:);
imagesc(time,1:numel(sorted_call_lengths),array(order,:) )
xlim(X_lim)
clim([-2 2])
axis xy
hold on
plot([0 0],[1 numel(sorted_call_lengths)], 'w')
plot(sorted_call_lengths,1:numel(sorted_call_lengths), 'w')
title(animal_label{an})

subplot(2,1,2)

[~, ~, ci]  = ttest(array);
no_nan = ~any(isnan(ci));
fill([time(no_nan) fliplr(time(no_nan))], [ci(1,no_nan) fliplr(ci(2,no_nan))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
hold on
plot(time,mean(array, 'omitmissing'), 'k')



plot(time,stacked_mean, 'k:')
hold on
plot(time,mean(stacked_mean), 'k:')
xlim(X_lim)

sig_idx = pvals_fdr < alpha;
limited_time = results.time;

borders = [ 0 sig_idx' 0];
beg_index = find(diff(borders)==1);

end_index = find(diff(borders)==-1)-1;

for bn = 1:numel(beg_index)
    time2plot = limited_time(beg_index(bn):end_index(bn))';
    fill([time2plot fliplr(time2plot)], [time2plot*0+fill_lim(1) time2plot*0+fill_lim(2)], 'r', 'EdgeColor','none', 'FaceAlpha',.2)
end

hold on
y_lim =ylim;
plot([0 0],y_lim, 'k' )
ylim tight

yyaxis right
plot(time, mean(all_onset_regressors( length_bool,:)), 'r')
xlim(X_lim)

%% save figure
print(gcf,'-vector','-dsvg',[figure_dir,'\calls and theta.svg'])
print(gcf,'-vector','-dsvg',[figure_dir,'\calls and theta direct_p_val.svg'])


%% load cvResults (to esitmate contribution of each varaible)
load([saving_folder,'\cvResults_mean_calls_theta_v2_AlllVar.mat'],'cvResults')
k = cvResults.k;
predictors = cvResults.predictors;

R2_full_allfolds =cvResults.Call.r2_full;
avgR2_full = mean(R2_full_allfolds);

y_for_sig = 20;
figure('units','normalized','outerposition',[.5 0 .5 1]);
subplot(2,1,1)
hold on
randx = .25*(rand(k,2)-.5);
re_order = [13 6 7 2 3 4 5 1 14 8 9 10 11 12]; % manual order
% re_order = 1:numel(predictors);
p_vals = nan(numel(re_order),1);

for j=1:numel(predictors)
    % Extract R² values for this predictor
    r2_full = cvResults.(predictors{re_order(j)}).r2_full;
    r2_reduced = cvResults.(predictors{re_order(j)}).r2_reduced;
    r2_diff = r2_full - r2_reduced; % difference per fold

    % Plot swarm of differences
    swarmchart((2*j-1)*ones(k,1), 100*r2_diff/avgR2_full, 'o', 'MarkerFaceColor','flat', 'MarkerFaceAlpha',.25);
    plot((2*j-1), 100*mean(r2_diff/avgR2_full), '_r','MarkerSize',10,'LineWidth',2);

    % Paired t-test (full vs reduced)
    % [~,p_val] = ttest(r2_diff);
    p_val = signrank(r2_diff);
    p_val = p_val * numel(predictors); % Bonferroni correction
    p_vals(re_order(j)) = p_val;

    % Annotate significance stars
    if p_val<0.001
        text(2*j -1, y_for_sig, '* * *','HorizontalAlignment','center')
    elseif p_val<0.01
        text(2*j -1, y_for_sig, ' * * ','HorizontalAlignment','center')
    elseif p_val<0.05
        text(2*j -1, y_for_sig, '  *  ','HorizontalAlignment','center')
    else
        text(2*j -1, y_for_sig, ' n.s ','HorizontalAlignment','center')
    end

    % Annotate actual p-value
    text(2*j -1, y_for_sig+ 5, num2str(p_val,'%.5f'), 'Rotation',45,'HorizontalAlignment','center')
end

% Reference line at 0
plot([0 2*numel(predictors)], [0 0], 'k')

% Formatting
xticks(1:2:2*numel(predictors))
xticklabels([])
ylabel('\Delta R^2 (Full - Reduced)')
ylim([-1 y_for_sig+10])
title('Cross-validated \DeltaR^2 when removing each predictor')
set(gca, 'FontSize', 24)


subplot(2,1,2)




hold on



for j=1:numel(predictors)
    % 
    betha_vals = cvResults.bethas_fulls_allfolds(:,re_order(j)+1); % difference per fold

    % Plot swarm of differences
    swarmchart((2*j-1)*ones(numel(betha_vals),1), betha_vals, 'o', 'MarkerFaceColor','flat', 'MarkerFaceAlpha',.25);
    plot((2*j-1), mean(betha_vals), '_r','MarkerSize',10,'LineWidth',2);
  
    % Paired t-test (full vs reduced)
    % [~,p_val] = ttest(r2_diff);
 
   
    
    % Annotate actual p-value
    % text(2*j -1, y_for_sig+ 5, num2str(p_val,'%.3f'), 'Rotation',45,'HorizontalAlignment','center')
end
plot([0 2*numel(predictors)], [0 0], 'k')

% Formatting
xticks(1:2:2*numel(predictors))
xticklabels(predictors(re_order))
set(gca, 'FontSize', 24)

%% save figure

print(gcf,'-vector','-dsvg',[figure_dir,'\varaible contribution theta.svg'])

%% LOAD DATA TO compare theta response to enganged and unenaged states

saving_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\Theta psth';

load([saving_folder,'\psth_structure_all_regressors_theta_hmm.mat']);
load([saving_folder,'\animal_names_all_regressorss_theta_hmm.mat'],'animal_names');

%%   stack data


% List of array variables to stack
varNames = { ...
    'state_onset', 'state_offset', ...
    'state_onset_regressor', 'state_offset_regressor',...
    'play_bout_onset_regressor', 'play_bout_offset_regressor', ...
    'call_onset_regressor', 'call_offset_regressor', ...
    'self_speed_onset_regressor', 'self_speed_offset_regressor', ...
    'self_acc_onset_regressor', 'self_acc_offset_regressor', ...
    'other_speed_onset_regressor', 'other_speed_offset_regressor', ...
    'animal_distance_onset_regressor', 'animal_distance_offset_regressor', ...
    'other_acc_onset_regressor', 'other_acc_offset_regressor', ...
    'self_onset_regressor', 'self_offset_regressor', ...
    'other_onset_regressor', 'other_offset_regressor', ...
    'play_bouts_table', 'hmm_onset_offset','hmm_type',...
    'time_wrapped','play_bout_latency' };
vars2zscore = {'self_speed_onset_regressor','self_speed_offset_regressor',...
    'animal_distance_onset_regressor', 'animal_distance_offset_regressor', ...
    'self_acc_onset_regressor','self_acc_offset_regressor',...
    'other_speed_onset_regressor', 'other_speed_offset_regressor', ...
    'other_acc_onset_regressor', 'other_acc_offset_regressor'};

% Initialize stacking container
stacked_data = struct();
for v = 1:numel(varNames)
    stacked_data.(varNames{v}) = [];
end
stacked_data.meta = {};  % {animal_name, partner_id, channel, play_length}

% Loop through each session
for i = 1:numel(psth_structure)
    S = psth_structure(i);

    % Get session info from parallel cell array
    animal_name = animal_names{i,1};
    partner_id  = animal_names{i,2};
    ch          = animal_names{i,3};

    % Compute play lengths
    hmm_lengths = S.hmm_onset_offset(:,2) - S.hmm_onset_offset(:,1);
    nRows = size(S.state_onset,1);

    % Stack all variables
    for v = 1:numel(varNames)

        if ismember(varNames{v}, vars2zscore)
            array = S.(varNames{v});
            array = (array - repmat(mean(S.(varNames{v})(:), 'omitmissing'), size(array,1), size(array,2)))./repmat(std(S.(varNames{v})(:), 'omitmissing'), size(array,1), size(array,2));
            stacked_data.(varNames{v}) = [stacked_data.(varNames{v}); array];
        else
            stacked_data.(varNames{v}) = [stacked_data.(varNames{v}); S.(varNames{v})];
        end
    end

    % Add metadata rows
    session_meta = [ ...
        repmat({animal_name}, nRows, 1), ...
        num2cell(repmat(partner_id, nRows, 1)), ...
        num2cell(repmat(ch, nRows, 1)), ...
        num2cell(hmm_lengths) ...
        ];
    stacked_data.meta = [stacked_data.meta; session_meta];
end


%% load  paired lengths for hmm with and without play

hmm_type            = stacked_data.hmm_type;
hmm_lengths = [stacked_data.meta{:,4} ];
is_there_play       =  hmm_type==1;
there_is_no_play    = hmm_type==0;

play_lengths        = hmm_lengths(is_there_play);
noplay_lengths      = hmm_lengths(there_is_no_play);

load([saving_folder,'\paired_lengths_all_regressorss_theta_hmm.mat'],'M', 'play_lengths', 'noplay_lengths');

%%  plot hmm state onset with and without play
x_lim = [-.5 2];
expanded_regressor = expand_half_intervals(stacked_data.state_onset_regressor);
bin_size =  psth_structure(1).wind_length -  psth_structure(1).wind_overlap;
time = psth_structure(1).hist_range(1):bin_size:psth_structure(1).hist_range(2);
baseline_index = time<0;

hmm_lengths                 = [stacked_data.meta{:,4}]';
hmm_type                    = stacked_data.hmm_type;
[sorted_play_bout_length, order_index] = sort(hmm_lengths);
sorted_hmm_type             = hmm_type(order_index)==1;
pow_matrix = stacked_data.state_onset;
current_hmm = stacked_data.state_onset_regressor;
only_baseline_effect =  expanded_regressor;





for j=1:size(pow_matrix,1)
    current_hmm(j,time<0) = 0;
    only_baseline_effect(j,time>hmm_lengths(j)) = 0;
    current_hmm(j,time>hmm_lengths(j)) = 0;
    pow_matrix(j,:) = (pow_matrix(j,:) - mean(pow_matrix(j, baseline_index), 'omitmissing'))/std(pow_matrix(j, baseline_index), 'omitmissing');
    pow_matrix(j,:)  = movmean( pow_matrix(j,:),.125/bin_size, 'omitmissing');
end





array = pow_matrix;

array(only_baseline_effect==0) = NaN;

array4mixed_model = array;

ci_state_play       = nan(2,size(pow_matrix,2));
ci_state_no_play    = nan(2,size(pow_matrix,2));
index_play = find(hmm_type==1);
index_play = index_play(M(:,1));

index_noplay = find(hmm_type==0);
index_noplay = index_noplay(M(:,2));
for t=1:size(pow_matrix,2)


    [~, ~, ci] = ttest(array(index_play,t));
    ci_state_play(:,t) = ci;

    [~, ~, ci] = ttest(array(index_noplay,t));
    ci_state_no_play(:,t) = ci;
end


figure
subplot(5,2,[1 3 5])
array_play = array(index_play,:);
lenght_play = hmm_lengths(index_play);
[sorted_length_play, order_play] = sort(lenght_play);
pcolor(time, 1:numel(sorted_length_play), array_play(order_play,:))
shading flat
hold on
plot((1:sum(sorted_hmm_type))*0, 1:numel(sorted_play_bout_length(sorted_hmm_type)), 'w')
plot(sorted_play_bout_length(sorted_hmm_type), 1:numel(sorted_play_bout_length(sorted_hmm_type)), 'w')
axis xy
xlim(x_lim)
clim([-1 2])
subplot(5,2,[7 9])
plot(time, mean(array_play, 'omitmissing'))
xlim(x_lim)


subplot(5,2,[1 3 5]+1)
array_noplay = array(index_noplay,:);
lenght_play = hmm_lengths(index_noplay);
[sorted_length_noplay, order_noplay] = sort(lenght_play);
pcolor(time, 1:numel(sorted_length_noplay), array_noplay(order_noplay,:))
shading flat
hold on
plot((1:sum(~sorted_hmm_type))*0, 1:numel(sorted_play_bout_length(~sorted_hmm_type)), 'w')
plot(sorted_play_bout_length(~sorted_hmm_type), 1:numel(sorted_play_bout_length(~sorted_hmm_type)), 'w')
axis xy
xlim(x_lim)
clim([-1 2])
subplot(5,2,[7 9]+1)
plot(time, mean(array_noplay, 'omitmissing'))
xlim(x_lim)

%% only play mean and confident intervals


hmm_figure = figure;
no_nan = ~any(isnan(ci_state_play));
fill([time(no_nan), fliplr(time(no_nan))], [ci_state_play(1, no_nan), fliplr(ci_state_play(2, no_nan))], 'r', 'FaceAlpha',.25, 'EdgeColor','none')
hold on
plot(time, mean(array_play, 'omitmissing'), 'r')


no_nan = ~any(isnan(ci_state_no_play));
fill([time(no_nan), fliplr(time(no_nan))], [ci_state_no_play(1, no_nan), fliplr(ci_state_no_play(2, no_nan))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
hold on
plot(time, mean(array_noplay, 'omitmissing'), 'k')
xlim(x_lim)
ylim tight

y_lim = ylim;


%% load cluster statistics for previous comparisson


alpha = 0.05;
relevan_time = [-.25 2];
time_index = time>=relevan_time(1) & time<=relevan_time(2);

load([saving_folder,'\cluster_pvals_fixed_length.mat'],'cluster_pvals','clusters','cluster_stats','perm_max_stats', 'array_play', 'array_noplay','alpha', 'time','time_index','relevan_time')
%% plot statistics
% Highlight significant clusters
sig_clusters = find(cluster_pvals < alpha);
sub_time= time(time_index);
figure(hmm_figure)
ylims = y_lim;
for i = 1:length(sig_clusters)
    cluster_idx = clusters{sig_clusters(i)};
    % Draw a patch (shaded area) over the cluster time points
    patch([sub_time(cluster_idx(1)) sub_time(cluster_idx(end)) sub_time(cluster_idx(end)) sub_time(cluster_idx(1))], ...
        [ylims(1) ylims(1) ylims(2) ylims(2)], ...
        [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
end

%% now ploting timewrapped data

%% ploting time wrapped
hmm_type = stacked_data.hmm_type;
wrapped_bins = 100; % hard coded in function
bin_size =  psth_structure(1).wind_length -  psth_structure(1).wind_overlap;
time = psth_structure(1).hist_range(1):bin_size:psth_structure(1).hist_range(2);
pre_index = 1:round(-psth_structure(1).hist_range(1)/bin_size);
pre_time =  (pre_index+psth_structure(1).hist_range(1)/bin_size)*bin_size;
hmm_lengths = [stacked_data.meta{:,4}]';
[sorted_play_bout_length, order] = sort(hmm_lengths);

wrapped_time_between = linspace(1/wrapped_bins,1,wrapped_bins);
wrapped_time_during = linspace(1 +1/wrapped_bins,2,wrapped_bins);

wrapped_time = [pre_time,wrapped_time_between,wrapped_time_during];


wrapped_bins = 100; %% had coded in fucntion
pow_matrix_tw = stacked_data.time_wrapped;
baseline_index_tw = wrapped_time<0;


for j=1:size(pow_matrix_tw,1)
    pow_matrix_tw(j,:) = (pow_matrix_tw(j,:) - mean(pow_matrix_tw(j, baseline_index_tw), 'omitmissing'))/std(pow_matrix_tw(j, baseline_index_tw), 'omitmissing');
    pow_matrix_tw(j,:)  = movmean( pow_matrix_tw(j,:),.125/bin_size, 'omitmissing');
end


index_all_conditions = ~any(isnan(pow_matrix_tw(:,wrapped_time>=0 )),2);
index_all_conditions =  stacked_data.play_bout_latency(:,1) <stacked_data.play_bout_latency(:,2);
% index_all_conditions = hmm_type ==1;

figure
subplot(5,1,1:2)
matrix2plot_with_play = pow_matrix_tw(index_all_conditions,:);
hmm_lenght_all_conditions = hmm_lengths(index_all_conditions);
[sorted_play_bout_length_all_cond, order_all_cond] = sort(hmm_lenght_all_conditions);
imagesc(wrapped_time, 1:size(matrix2plot_with_play,1), matrix2plot_with_play(order_all_cond,:))
xlim([-2 2])
clim([-2 3])
axis xy

subplot(5,1,3:5)
index_all_conditions = stacked_data.play_bout_latency(:,1) >stacked_data.play_bout_latency(:,2);
matrix2plot_without_play = pow_matrix_tw(index_all_conditions,:);


[~, ~, ci] = ttest(matrix2plot_without_play);
no_nan = ~any(isnan(ci));

fill([wrapped_time(no_nan), fliplr(wrapped_time(no_nan))], [ci(1,no_nan) fliplr(ci(2, no_nan))], 'k', 'FaceAlpha', .25, 'EdgeColor', 'none')
hold on
plot(wrapped_time, mean(matrix2plot_without_play, 'omitmissing'), 'k')
xlim([-2 2])


[~, ~, ci] = ttest(matrix2plot_with_play);
no_nan = ~any(isnan(ci));

fill([wrapped_time, fliplr(wrapped_time)], [ci(1,no_nan) fliplr(ci(2, no_nan))], 'r', 'FaceAlpha', .25, 'EdgeColor', 'none')
hold on
plot(wrapped_time, mean(matrix2plot_with_play, 'omitmissing'), 'r')
xlim([-2 2])



%% load clustering stastistics


alpha = 0.05;
relevan_time = [-.2 1];
time_index = wrapped_time>=relevan_time(1) & wrapped_time<=relevan_time(2);

load([saving_folder,'\cluster_pvals_hmm_stats.mat'],'cluster_pvals','clusters','cluster_stats','perm_max_stats', 'matrix2plot_with_play', 'matrix2plot_without_play','alpha', 'time','time_index','relevan_time')




%% plotin time wrapped data and significant differneces
alpha_level = 0.05;
figure

index_all_conditions = stacked_data.play_bout_latency(:,1) >stacked_data.play_bout_latency(:,2);
% matrix2plot_without_play = pow_matrix_tw(index_all_conditions,:);


[~, ~, ci] = ttest(matrix2plot_without_play);
no_nan = ~any(isnan(ci));
sub_time = wrapped_time(time_index);
fill([wrapped_time(no_nan), fliplr(wrapped_time(no_nan))], [ci(1,no_nan) fliplr(ci(2, no_nan))], 'k', 'FaceAlpha', .25, 'EdgeColor', 'none')
hold on
plot(wrapped_time, mean(matrix2plot_without_play, 'omitmissing'), 'k')
xlim([-2 2])


[~, ~, ci] = ttest(matrix2plot_with_play);
no_nan = ~any(isnan(ci));

fill([wrapped_time, fliplr(wrapped_time)], [ci(1,no_nan) fliplr(ci(2, no_nan))], 'r', 'FaceAlpha', .25, 'EdgeColor', 'none')
hold on
plot(wrapped_time, mean(matrix2plot_with_play, 'omitmissing'), 'r')
xlim([-2 2])

ylims = ylim;

for i = 1:length(sig_clusters)
    cluster_idx = clusters{sig_clusters(i)};
    % Draw a patch (shaded area) over the cluster time points
    patch([sub_time(cluster_idx(1)) sub_time(cluster_idx(end)) sub_time(cluster_idx(end)) sub_time(cluster_idx(1))], ...
        [ylims(1) ylims(1) ylims(2) ylims(2)], ...
        [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
end

[h,p,stats] = ttest2(matrix2plot_without_play,matrix2plot_with_play);


p = p(time_index);

[L,n] =bwlabeln(p<alpha_level);
start_vals = find(diff([ 0 p<0.05 0])==1);
end_vals = find(diff([ 0 p<0.05 0])==-1)-1;

for i = 1:numel(start_vals)
    % Draw a patch (shaded area) over the cluster time points
    patch([sub_time(start_vals(i)) sub_time(end_vals(i)) sub_time(end_vals(i)) sub_time(start_vals(i))], ...
        [ylims(1) ylims(1) ylims(2) ylims(2)], ...
        [0.9 0 0], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
end

%%


print(gcf,'-vector','-dsvg',[figure_dir,'\enganged and unengaged time wrapped.svg'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% DELTA SECTION NOW %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% LOAD DELTA DATA NOW
%% LOAD DELTA DATA NOW
%% LOAD DELTA DATA NOW
%% LOAD DELTA DATA NOW

disp('loading')
load([saving_folder,'\psth_structure_delta.mat'],'psth_structure');
load([saving_folder,'\animal_names_delta.mat'],'animal_names');
%% mergind psth data 
smooth_wind = 20;
baseline_range = [-2 0]
animal_label = {'B1D1','B1S3','B2S2','B3D2', 'B4S2', 'B4D4'};
electorde_numner = [1 2];
bin_size = psth_structure(1).wind_length - psth_structure(1).wind_overlap;
psth_ranges = psth_structure(1).hist_range;
wrap_range = psth_structure(1).range_time_wrap;
time = psth_ranges(1):bin_size:psth_ranges(2)+bin_size;
baseline_index = time<baseline_range(2) & time>baseline_range(1);
baseline_index_time_wrap = 1:round((abs(wrap_range(1))/bin_size));
all_psth_onset          = [];
all_psth_onset_behavior = [];
all_psth_onset_only_playobut    = [];

all_psth_offset         = [];
all_psth_tw             = [];
all_psth_tw_3points     = [];
all_play_bouts          = [];
time_wrap_time          = [(baseline_index_time_wrap*bin_size) + wrap_range(1),linspace(0,1,psth_structure(1).n_bins_time_wrap),1 + (1:round((abs(wrap_range(2))/bin_size)))*bin_size];
time_wrap_3_points      = [(baseline_index_time_wrap*bin_size) + wrap_range(1),linspace(0,1-1/psth_structure(1).n_bins_time_wrap,psth_structure(1).n_bins_time_wrap), ...
    linspace(1,2-1/psth_structure(1).n_bins_time_wrap,psth_structure(1).n_bins_time_wrap),2 + (1:round((abs(wrap_range(2))/bin_size)))*bin_size];


animal_index = [];
electrode_index = []
for j=1:numel(psth_structure)

    if contains(animal_names{j},animal_label)

        animal_num      = find(cell2mat(cellfun(@(x) contains(animal_names{j},x), animal_label, 'UniformOutput',false)));
        electrode_num   = animal_names{j,2}  ;
        this_animal_playbouts = psth_structure(j).play_bouts_table;
        this_animal_lengths = diff(this_animal_playbouts');

        this_psth_onset         = psth_structure(j).play_bout_onset;
        this_psth_onset_onlypb  = this_psth_onset;

        animal_index = [animal_index;repmat(animal_num,size(this_psth_onset,1),1)];
        electrode_index = [electrode_index;repmat(electrode_num,size(this_psth_onset,1),1)];

        for trial=1:size(this_psth_onset,1)
            this_psth_onset(trial,:) = ( this_psth_onset(trial,:) - mean( this_psth_onset(trial,baseline_index)))/std( this_psth_onset(trial,baseline_index));
            this_psth_onset(trial,:) = movmean(this_psth_onset(trial,:), smooth_wind);
            this_psth_onset_onlypb(trial,:) = this_psth_onset(trial,:);
            this_psth_onset(trial,time> this_animal_lengths(trial)) = NaN;
        end
        all_psth_onset      = [all_psth_onset; this_psth_onset];
        all_psth_onset_only_playobut = [all_psth_onset_only_playobut; this_psth_onset_onlypb];

        this_psth_onset     = psth_structure(j).play_bout_onset;
        this_psth_offset    = psth_structure(j).play_bout_offset;
        for trial=1:size(this_psth_offset,1)
            this_psth_offset(trial,:) = ( this_psth_offset(trial,:) - mean( this_psth_onset(trial,baseline_index)))/std( this_psth_onset(trial,baseline_index));
            this_psth_offset(trial,:) = movmean(this_psth_offset(trial,:), smooth_wind);
        end
        all_psth_offset = [all_psth_offset; this_psth_offset];

        this_psth_tw = psth_structure(j).play_bout_tw_this;
        for trial=1:size(this_psth_tw,1)
            this_psth_tw(trial,:) = ( this_psth_tw(trial,:) - mean( this_psth_tw(trial,baseline_index_time_wrap)))/std( this_psth_tw(trial,baseline_index_time_wrap));
        end
        all_psth_tw = [all_psth_tw; this_psth_tw];


        this_psth_tw = psth_structure(j).three_point_tw;
        for trial=1:size(this_psth_tw,1)
            this_psth_tw(trial,:) = ( this_psth_tw(trial,:) - mean( this_psth_tw(trial,baseline_index_time_wrap)))/std( this_psth_tw(trial,baseline_index_time_wrap));
        end
        all_psth_tw_3points = [all_psth_tw_3points; this_psth_tw];



        this_psth_ab = psth_structure(j).animal_behavior_onset;
        for trial=1:size(this_psth_ab,1)
            this_psth_ab(trial,:) = ( this_psth_ab(trial,:) - mean( this_psth_ab(trial,baseline_index_time_wrap)))/std( this_psth_ab(trial,baseline_index_time_wrap));
            this_psth_ab(trial,:) = movmean(this_psth_ab(trial,:), smooth_wind);
        end
        all_psth_onset_behavior = [all_psth_onset_behavior; this_psth_ab];

        all_play_bouts = [all_play_bouts;this_animal_playbouts];
    end
end

play_bout_length = diff(all_play_bouts')';


[sorted_play_bout_length, order] = sort(play_bout_length);
%% plot single animals  (and obtain mean response per animal)
X_lim = [-2 3]
figure
min_length = .0;
stacked_mean_onset = [];
% what2plot = all_psth_onset_behavior; %Select what to plot
% % what2plot = all_psth_onset;
what2plot = all_psth_tw_3points;
zscore_limit = 4;
% time2use = time;
time2use = time_wrap_3_points;

artifcat_removal = max(abs(what2plot(:,time2use>X_lim(1))),[],2,'omitmissing')<4;
for an= 1:numel(animal_label)
    animal_bool = animal_index==an;
    length_bool = play_bout_length>min_length;
    electrode_bool = electrode_index==1 & artifcat_removal;
    [sorted_play_bout_length, order] = sort(play_bout_length(animal_bool & length_bool & electrode_bool,:));
    subplot(5,numel(animal_label),(1:numel(animal_label):2*numel(animal_label)) + an-1)

    array = what2plot(animal_bool & length_bool & electrode_bool,:);
    imagesc(time2use,1:numel(sorted_play_bout_length),array(order,:) )
    xlim(X_lim)
    clim([-2 2])
    axis xy
    hold on
    plot([0 0],[1 numel(sorted_play_bout_length)], 'w')
    plot(sorted_play_bout_length,1:numel(sorted_play_bout_length), 'w')
    title(animal_label{an})

    subplot(5,numel(animal_label),((2*numel(animal_label) + 1):numel(animal_label):5*numel(animal_label)) + an-1)

    [~, ~, ci]  = ttest(array);
    no_nan = ~any(isnan(ci));
    fill([time2use(no_nan) fliplr(time2use(no_nan))], [ci(1,no_nan) fliplr(ci(2,no_nan))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time2use,mean(array, 'omitmissing'), 'k')
    xlim(X_lim)
    stacked_mean_onset = [stacked_mean_onset;mean(array, 'omitmissing')];
end

%% load mixed model effect

% load([saving_folder,'\results_play_bout_delta_all_onlybout.mat'],'results'); %using only time points within playbouts
% load([saving_folder,'\results_play_bout_onlybout_zscore4.mat'],'results') % removing trials with a max zscore during playbout of 4, to prevent outliers
% load([saving_folder,'\results_play_bout_onlybehavior_zscore4.mat'],'results');
load([saving_folder,'\results_play_bout_tw3points_zscore4.mat'],'results');
% 

limited_time = results.time;
est = results.est;
ci = results.ci;
pvals_fdr = results.pvals;
% pvals_fdr = results.pvals_fdr;
%% plot all together

X_lim = [-2 3];
alpha = 0.05;
figure
min_length = .0;
fill_lim = [.35 .4];

length_bool = play_bout_length>min_length ;
electrode_bool = electrode_index==1;
[sorted_play_bout_length, order] = sort(play_bout_length( length_bool & electrode_bool & artifcat_removal,:));

subplot(2,1,1)
array = what2plot( length_bool & electrode_bool & artifcat_removal,:);
imagesc(time2use,1:numel(sorted_play_bout_length),array(order,:) )

xlim(X_lim)
clim([-1 2])
axis xy
hold on
plot([0 0],[1 numel(sorted_play_bout_length)], 'w')
plot(sorted_play_bout_length,1:numel(sorted_play_bout_length), 'w')
title('All data')

subplot(2,1,2)

% [~, ~, ci]  = ttest(array);
% no_nan = ~any(isnan(ci));
% fill([time2use(no_nan) fliplr(time2use(no_nan))], [ci(1,no_nan) fliplr(ci(2,no_nan))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
% hold on
% plot(time2use,mean(array, 'omitmissing'), 'k')
% xlim(X_lim)
plot(time2use,stacked_mean_onset, 'k:')
hold on
% plot(time2use,mean(stacked_mean_onset), 'k:')
limited_time = results.time';
ci = results.ci';
 no_nan = ~any(isnan(ci));
fill([limited_time(no_nan) fliplr(limited_time(no_nan))], [ci(1,no_nan) fliplr(ci(2,no_nan))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
xlim(X_lim)

sig_idx = pvals_fdr < alpha;

%%
limited_time = results.time;

borders = [ 0 sig_idx' 0];
beg_index = find(diff(borders)==1);

end_index = find(diff(borders)==-1)-1;

for bn = 1:numel(beg_index)
    time2plot = limited_time(beg_index(bn):end_index(bn))';
    fill([time2plot fliplr(time2plot)], [time2plot*0+fill_lim(1) time2plot*0+fill_lim(2)], 'r', 'EdgeColor','none')
end

hold on
y_lim =ylim;
plot([0 0],y_lim, 'k' )
ylim tight

%% save figure
print(gcf,'-vector','-dsvg',[figure_dir,'\play and delta only playoubt.svg'])


%% load now chage in explained variane by variable
% predictors = {'Call','SelfSpeed','OtherSpeed','OtherAcc','SelfAcc','PlayBout'};
% load([saving_folder,'\cvResults_single_calls.mat'],'cvResults')
load([saving_folder,'\cvResults_mean_calls_delta_v2_AlllVar.mat'],'cvResults')
k = cvResults.k;
predictors = cvResults.predictors;

R2_full_allfolds =cvResults.Call.r2_full;
avgR2_full = mean(R2_full_allfolds);

y_for_sig = 20;
figure('units','normalized','outerposition',[.5 0 .5 1]);
subplot(2,1,1)
hold on
randx = .25*(rand(k,2)-.5);
re_order = [13 6 7 2 3 4 5 1 14 8 9 10 11 12]; % manual order
% re_order = 1:numel(predictors);
p_vals = nan(numel(re_order),1);

for j=1:numel(predictors)
    % Extract R² values for this predictor
    r2_full = cvResults.(predictors{re_order(j)}).r2_full;
    r2_reduced = cvResults.(predictors{re_order(j)}).r2_reduced;
    r2_diff = r2_full - r2_reduced; % difference per fold

    % Plot swarm of differences
    swarmchart((2*j-1)*ones(k,1), 100*r2_diff/avgR2_full, 'o', 'MarkerFaceColor','flat', 'MarkerFaceAlpha',.25);
    plot((2*j-1), 100*mean(r2_diff/avgR2_full), '_r','MarkerSize',10,'LineWidth',2);

    % Paired t-test (full vs reduced)
    % [~,p_val] = ttest(r2_diff);
    p_val = signrank(r2_diff);
    p_val = p_val * numel(predictors); % Bonferroni correction
    p_vals(re_order(j)) = p_val;

    % Annotate significance stars
    if p_val<0.001
        text(2*j -1, y_for_sig, '* * *','HorizontalAlignment','center')
    elseif p_val<0.01
        text(2*j -1, y_for_sig, ' * * ','HorizontalAlignment','center')
    elseif p_val<0.05
        text(2*j -1, y_for_sig, '  *  ','HorizontalAlignment','center')
    else
        text(2*j -1, y_for_sig, ' n.s ','HorizontalAlignment','center')
    end

    % Annotate actual p-value
    text(2*j -1, y_for_sig+ 5, num2str(p_val,'%.5f'), 'Rotation',45,'HorizontalAlignment','center')
end

% Reference line at 0
plot([0 2*numel(predictors)], [0 0], 'k')

% Formatting
xticks(1:2:2*numel(predictors))
xticklabels([])
ylabel('\Delta R^2 (Full - Reduced)')
ylim([-1 y_for_sig+10])
title('Cross-validated \DeltaR^2 when removing each predictor')
set(gca, 'FontSize', 24)


subplot(2,1,2)




hold on



for j=1:numel(predictors)
    % 
    betha_vals = cvResults.bethas_fulls_allfolds(:,re_order(j)+1); % difference per fold

    % Plot swarm of differences
    swarmchart((2*j-1)*ones(numel(betha_vals),1), betha_vals, 'o', 'MarkerFaceColor','flat', 'MarkerFaceAlpha',.25);
    plot((2*j-1), mean(betha_vals), '_r','MarkerSize',10,'LineWidth',2);
  
    % Paired t-test (full vs reduced)
    % [~,p_val] = ttest(r2_diff);
 
   
    
    % Annotate actual p-value
    % text(2*j -1, y_for_sig+ 5, num2str(p_val,'%.3f'), 'Rotation',45,'HorizontalAlignment','center')
end
plot([0 2*numel(predictors)], [0 0], 'k')

% Formatting
xticks(1:2:2*numel(predictors))
xticklabels(predictors(re_order))
set(gca, 'FontSize', 24)

%% save figure

print(gcf,'-vector','-dsvg',[figure_dir,'\varaible contribution delta.svg'])

%% LOAD delta and distance data
load([saving_folder,'\psth_structure_speed_delta.mat'],'psth_structure')

load([saving_folder,'\animal_names_speed_delta.mat'],'animal_names')

%%

stasts_table = cell(numel(psth_structure),6);

for j=1:numel(psth_structure)

stasts_table(j,:) = { psth_structure(j).lm.Coefficients.Estimate(4), ...
                psth_structure(j).lm.Coefficients.pValue(4),...
                psth_structure(j).lm.Coefficients.tStat(4),...
                  animal_names{j,:}};

table = psth_structure(j).model_data;

end

stasts_table =cell2table(stasts_table);
stasts_table.Properties.VariableNames = {'Estimate','pValue','tStat','Animal','Partner','Electrode'};

stasts_table = stasts_table(stasts_table.Electrode==1,:);

lme = fitlme(stasts_table, 'Estimate ~ 1 + (1|Animal)');
coef = fixedEffects(lme);       % estimated mean
ci = coefCI(lme);               % confidence interval
pval = lme.Coefficients.pValue; % p-value for intercept

fprintf('Mean = %.3f, 95%% CI [%.3f, %.3f], p = %.4f\n', coef, ci(1), ci(2), pval);

%% fix parameters
n_grid = 101;
distance_grid = linspace(-2, 4, n_grid);

%% estiamte bined difference for play and no play at fixed distances
data_together = [];
mean_powers_play    = nan(numel(psth_structure) ,n_grid);
mean_powers_noplay  = nan(numel(psth_structure) ,n_grid);
animal_names_column = {};
for j=1:numel(psth_structure) 


tbl = psth_structure(j).model_data;
tbl = tbl(~isnan(tbl.Speed),:);

Y = tbl.Power;
Y = (Y - mean(Y, 'omitmissing'))/std(Y, 'omitmissing');
X = tbl.Distance;
Xa = X(tbl.Play=='true');
Ya = Y(tbl.Play=='true');


Xb = X(tbl.Play=='false');
Yb =  Y(tbl.Play=='false');

animal_names_column = [animal_names_column;repmat(animal_names(j,1),numel(Y),1)];
data_together = [data_together;[X Y ]];

edges = [distance_grid Inf];

% Assign each sample to a bin
[~,~,binA] = histcounts(Xa, edges);
[~,~,binB] = histcounts(Xb, edges);

% Pre-allocate as NaN
meanA = nan(size(distance_grid));
meanB = nan(size(distance_grid));

% Compute mean per bin (only for bins with data)
for i = 1:numel(distance_grid)
    meanA(i) = mean(Ya(binA == i), 'omitnan');
    meanB(i) = mean(Yb(binB == i), 'omitnan');
end


mean_powers_play(j,:)    = meanA;
mean_powers_noplay(j,:)  = meanB;
end
%% plot distnace to power rleation for delta
[bin_count, ~, samples2plot] = histcounts(data_together(:,1), -2:.1:4);

figure
bins_with_values = unique(samples2plot)';
max_count = 100;
min_count = 20;

indexes2plot = [];

for j=bins_with_values
    indexes= find(samples2plot==j);
   if numel(indexes)>=max_count

       indexes  = datasample(indexes,max_count );
   end
   if numel(indexes)>=min_count
       indexes2plot = [indexes2plot;indexes];
   end
end
selection_index = ~any(isnan(data_together(indexes2plot,:)),2)  & data_together(indexes2plot,1)<2.5 & abs(data_together(indexes2plot,2))<3;


subplot(1,2,1)
plot(data_together(indexes2plot(selection_index),1), data_together(indexes2plot(selection_index),2), '.k', 'MarkerSize',    .1)
[c,p] = corr(data_together(indexes2plot(selection_index),1),data_together(indexes2plot(selection_index),2));
title(num2str([c,p]))
xlim([-2 3])
axis xy
table2corr = array2table(data_together);
table2corr.Properties.VariableNames = {'Distance','Power'};
table2corr.Animal =categorical(animal_names_column); 
 lme = fitlme(table2corr,'Power ~ Distance + (1|Animal)');

slope_dist      = lme.Coefficients.Estimate(2);
dist_intercept = lme.Coefficients.Estimate(1);

x_spaced = distance_grid;
hold on
plot(x_spaced, x_spaced*slope_dist + dist_intercept, 'r')
subplot(1,2,2)
plot(distance_grid, mean_powers_play-mean_powers_noplay, ':k');
hold on
mean_diff =  mean(mean_powers_play-mean_powers_noplay);
[~, ~, ci] = ttest(mean_powers_play-mean_powers_noplay);

no_nan =~any(isnan(ci));
fill([distance_grid(no_nan) fliplr(distance_grid(no_nan))], [ci(1,no_nan) fliplr(ci(2,no_nan))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')

plot(distance_grid,mean_diff, 'k');
xlim([-1 2])
ylim([-1.5 1.5])


%% save figure

print(gcf,'-vector','-dsvg',[figure_dir,'\distance and delta power relation.svg'])
%% 
%%
%% DELTA-THETA COUPLING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% DELTA-THETA COUPLING %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% load if needed
disp('loading')
load([saving_folder,'\psth_structure_delta.mat'],'psth_structure');
load([saving_folder,'\animal_names_delta.mat'],'animal_names');
disp('ready')



%% merging

all_mean_AMP    = [];
all_mean_norm   = [];
all_real_MI     = [];
all_shuffled_MI = [];
all_real_r     = [];
all_shuffled_r = [];
for ns =1:numel(psth_structure)

all_mean_AMP    = [all_mean_AMP;psth_structure(ns).meanAmp_real];
all_mean_norm   = [all_mean_norm;zscore(psth_structure(ns).meanAmp_real)];

this_session_shuffled_MI    = psth_structure(ns).MI_perm;
this_session_real_MI        = psth_structure(ns).MI_real;
this_session_real_MI        = (this_session_real_MI - mean(this_session_shuffled_MI))/std(this_session_shuffled_MI);
this_session_shuffled_MI    = zscore(this_session_shuffled_MI);
all_shuffled_MI             = [all_shuffled_MI;this_session_shuffled_MI];
all_real_MI                 = [all_real_MI;this_session_real_MI];


this_session_shuffled_r     = psth_structure(ns).r_perm;
this_session_real_r         = psth_structure(ns).r_real;
this_session_real_r         = (this_session_real_r - mean(this_session_shuffled_r))/std(this_session_shuffled_r);
this_session_shuffled_r     = zscore(this_session_shuffled_r);
all_real_r                  = [all_real_r;this_session_real_r];
all_shuffled_r              = [all_shuffled_r;this_session_shuffled_r];


end

%%
nBins = 18;
electrode_index = ismember([animal_names{:,2}],1);
 edges = linspace(-pi, pi, nBins);
 zero_deg_mod = mean(all_mean_norm(:,edges>= -.5 & edges<=.5 ),2);
 [~, mod_order] = sort(zero_deg_mod);
 electrode_index = electrode_index(mod_order);
figure_figure =  figure('units','normalized','outerposition',[0 .5 1 .5]);
subplot(1,5,1)
imagesc(180*edges/pi, 1:sum(electrode_index),all_mean_norm(mod_order(electrode_index),:))
axis xy
yticks(1:sum(electrode_index))
yticklabels(animal_names(mod_order(electrode_index),1))

subplot(1,5,2)
plot(180*edges/pi, all_mean_norm(zero_deg_mod>0,:), 'r:')
hold on
plot(180*edges/pi, mean(all_mean_norm(zero_deg_mod>0,:)), 'r', 'LineWidth',2)

plot(180*edges/pi, all_mean_norm(zero_deg_mod<0,:), 'b:')
hold on
plot(180*edges/pi, mean(all_mean_norm(zero_deg_mod<0,:)), 'b', 'LineWidth',2)


subplot(1,5,3)
hold off
histogram(all_shuffled_MI, 100, 'EdgeColor','none','FaceColor','k', 'Normalization','percentage')
xscale log
ylim([0 4])
yyaxis right
hold on
histogram(all_real_MI, numel(all_real_MI), 'EdgeColor','none','FaceColor','r')
xticks([1 10 100 1000 10000])
ylim([0 4])




%% load single session for correlation example example

fn =4;
current_dir = [npx_Raw_Data, '\', animal_list(fn).name];
area_limit_table    = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\Area_limits_GoodLooking.xlsx';
% npx_raw_data = 
animal_code         = strsplit(current_dir, '\');
animal_code         = animal_code{end};
animal_code_params  = strsplit(animal_code, ' ');
animal_batch        = animal_code_params{1};
repeated_animal     = animal_code_params{3};

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
%% obtain LFP AND FILTER DATA



PAG_LFP         = double(LFP(mid_PAG_channel,:));
clear LFP
ch_n=1;
disp('Filtering Signal')

filtered_signal_freq1 = filtfilt(Hd_freq1.Coefficients, 1, PAG_LFP(ch_n,:));
hiblert_data1 = hilbert(filtered_signal_freq1);

filtered_signal_freq2 = filtfilt(Hd_freq2.Coefficients, 1, PAG_LFP(ch_n,:));
hiblert_data2 = hilbert(filtered_signal_freq2);


phase_data1         = angle(hiblert_data1);
amplitud_data1      = abs(hiblert_data1);
amplitud_data2      = abs(hiblert_data2);

%%
figure(figure_figure)
max_n_points = 200;
max_delta   = 1600;
points2plot = 1000;
% index = randsample(numel(amplitud_data1), points2plot);
n_bins = 100;
[counts, ~, bn_index] = histcounts(amplitud_data1,linspace(min(amplitud_data1), max_delta,n_bins+1));

valsperbin = min(round(.5*min(counts)),max_n_points);;

subsampled_data = nan(valsperbin*n_bins,1);

for j=1:n_bins
    subsampled_data((1 + valsperbin*(j-1)):(valsperbin*(j))) = randsample(find(bn_index==j),valsperbin);
end

data2corr = [amplitud_data1(subsampled_data)',amplitud_data2(subsampled_data)'];
plot(data2corr(:,1),data2corr(:,2), '.' )




central_values = abs(zscore(data2corr(:,1)))<2 & abs(zscore(data2corr(:,2)))<2;
subplot(1,5,4)
plot(data2corr(central_values,1),data2corr(central_values,2), 'k.', 'MarkerSize',.1 )
% [c,p] = corr(data2corr(central_values,:));
title(p(1,2))


subplot(1,5,5)
hold off
histogram(all_shuffled_r, 100, 'EdgeColor','none','FaceColor','k', 'Normalization','percentage')
xscale log
hold on
ylim([0 4])
yyaxis right
histogram(all_real_r, numel(all_real_r), 'EdgeColor','none','FaceColor','r')
xticks([0.1 1 10 100 1000 10000])
xlim tight
ylim([0 4])


%% save resulint figure
print(gcf,'-vector','-dsvg',[figure_dir,'\thet delta entreinment.svg'])