

hmm_directory       = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\HMM data\HMM raw data';
synch_directory     = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Synch data';
area_limit_table    = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\Area_limits_GoodLooking.xlsx';
current_dir         = cd;
animal_code         = strsplit(current_dir, '\');
animal_code         = animal_code{end};
animal_code_params  = strsplit(animal_code, ' ');
animal_batch        = animal_code_params{1};
date                = animal_code_params{2};
repeated_animal     = animal_code_params{3};

%% load synch from synch folder
cd([synch_directory, '\', animal_code])
load('synch_model_video2NPX')
load('synch_model_audio2NPX')
cd(current_dir)
current_dir = cd;
play_behaviors      = {'Pounce', 'CC','Boxing', 'Evasion','Pin', 'Escape', 'CB'};

%% 2 Load hmm data from HMM folder

cd([hmm_directory, '\',animal_code])
load([animal_code,' hmm_binned_time'])
load traking_structure
restrict2Partnerssession = true;

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

config.Behavior = Behavior;
config.repeated_animal =repeated_animal;
config.animal_types =animal_types        ;
config.play_behaviors =play_behaviors      ;
config.beh_bin =beh_bin             ;
config.conv_length =conv_length;
config.behavior_window = 0;

traking_time        = predict(synch_model_video2NPX, (traking_structure.frames2stract/30)')';
animal_pos          = nan(numel(hmm_binned_time),2);
partner_pos         = nan(numel(hmm_binned_time),2);
animal_pos(:,1) = interp1(traking_time, smoothdata(traking_structure.animal_pos(:,1), 'loess',5), hmm_binned_time,'cubic');
animal_pos(:,2) = interp1(traking_time,smoothdata(traking_structure.animal_pos(:,2), 'loess',5), hmm_binned_time,'cubic');


partner_pos(:,1) = interp1(traking_time, smoothdata(traking_structure.partner_pos(:,1), 'loess',5), hmm_binned_time,'cubic');
partner_pos(:,2) = interp1(traking_time,smoothdata(traking_structure.partner_pos(:,2), 'loess',5), hmm_binned_time,'cubic');




[play_bouts_table]  = play_bout(config);
hmm_binned_time     = predict(synch_model_audio2NPX,hmm_binned_time')';
play_bout_time      = any(hmm_binned_time>=play_bouts_table(:,1) & hmm_binned_time<=play_bouts_table(:,2))';
min_time2analysis   = Behavior.Start(ismember(Behavior.Type,'Partners session'))  ;
max_time2analysis   = Behavior.End(ismember(Behavior.Type,'Partners session'))  ;

time_limit          = (hmm_binned_time>= min_time2analysis & hmm_binned_time<=  max_time2analysis )';
Behavior.Start      = predict(synch_model_audio2NPX    , Behavior.Start);
Behavior.End        = predict(synch_model_audio2NPX, Behavior.End);

hmm_states =  readNPY( ['2states',animal_code,'.npy']);
hmm_3states =  readNPY(['3states',animal_code,'.npy']);

play_state = 0;
non_play_state  = 1;

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


A = hmm_3states';

% Find where the value changes
changePoints = [1, find(diff(A) ~= 0) + 1, length(A) + 1];

% Preallocate results
results = [];

% Loop over each segment
for i = 1:length(changePoints) - 1
    startIdx = changePoints(i);
    endIdx = changePoints(i+1) - 1;
    value = A(startIdx);
    results = [results; value, startIdx, endIdx];
end

beg_end_times_3states =[double(results(:,1)),  hmm_binned_time(results(:, [2 3]))];


%%  load lfp from current dir
cd(current_dir)
disp('LOADING LFP')
if exist([cd,'\','LFP_PAG.mat'], 'file')==2

    NPX_Type        = 2;
    load LFP_PAG
elseif exist([cd,'\','LFP_PAG.dat'], 'file')==2
    NPX_Type        = 1;
    file_pointer    = fopen('LFP_PAG.dat', 'r');
    LFP             = fread(file_pointer,'int16');
    LFP             = reshape(LFP, 384, numel(LFP)/384);
end


disp('LFP LOADED')
%% select_mid_pag_channel 
disp('Loadinbg Channel Map')
hard_coded_x_coords = [8 40;258 290; 508 540; 758 790];
area_limit = readtable(area_limit_table);
this_animal = ['Batch', animal_batch(2), repeated_animal,animal_batch(4)];
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
%%

% PAG_LFP         = double(LFP(mid_PAG_channel,:));
% clear LFP
LFP_time        = (1:size(PAG_LFP,2))/2500;
wind_length = .250;
wind_overlap = 0.240;
f = 4:.1:15;
theta_range = [6 12];

f_index = f>=theta_range(1) & f<=theta_range(2);
range2exctract                  = LFP_time>=min_time2analysis & LFP_time<=max_time2analysis;
 stack_lm_structure = [];
for ch_n=1:numel(mid_PAG_channel)
    disp('Estimating spectrogram')
    [pow_spectrogram,~,spect_time]  = spectrogram(PAG_LFP(ch_n,range2exctract),wind_length*2500, wind_overlap*2500, f,2500);
    spect_time                     = spect_time+min_time2analysis;

    pow_spectrogram = abs(pow_spectrogram);
    disp('ready')
   
    signal_in_range      = zscore(mean(log10(pow_spectrogram)))<3;

    theta_pow_spect_t =  mean(log10(pow_spectrogram(f_index,:)));
    theta_pow_spect_t   = movmean(theta_pow_spect_t,1/max(theta_range));


    theta_pow           = interp1(spect_time,theta_pow,hmm_binned_time);
    theta_pow_zscored = theta_pow;
    theta_pow_zscored(~isnan(theta_pow))    = zscore(theta_pow_zscored(~isnan(theta_pow)));
    theta_pow_in_range                      = abs(theta_pow_zscored)<3;
    animal_speed                            = sqrt(sum(diff(animal_pos).*diff(animal_pos),2));
    animal_acceleration                     = abs(diff(animal_speed));


    spect_t_speed           = interp1(hmm_binned_time(1:end-1),animal_speed ,spect_time);
    spect_t_speed_zscored   = spect_t_speed;
    spect_t_speed_zscored(~isnan(spect_t_speed_zscored))   = zscore(spect_t_speed_zscored(~isnan(spect_t_speed_zscored)));


    spect_t_acceleration    = interp1(hmm_binned_time(1:end-2),animal_acceleration ,spect_time);
    spect_t_acceleration_zscored   = spect_t_acceleration;
    spect_t_acceleration_zscored(~isnan(spect_t_acceleration_zscored))   = zscore(spect_t_acceleration_zscored(~isnan(spect_t_acceleration_zscored)));


    signal_in_range         = signal_in_range & abs(spect_t_speed_zscored)<3 & abs(spect_t_acceleration_zscored)<3;
    [~, speed_order]        =  sort(spect_t_speed);
    [~, acc_order]          =  sort(spect_t_acceleration);

    n_stacks = 30;

    speed_stack             = linspace(floor(min(spect_t_speed)), ceil(max(spect_t_speed)), n_stacks);
    acceleration_stack      = linspace(floor(min(spect_t_acceleration)), ceil(max(spect_t_acceleration)), n_stacks);

    speed_matrix            = nan(n_stacks,numel(f));
    acceleration_matrix     = nan(n_stacks,numel(f));

    for j=1:n_stacks-1
        speed_matrix(j,: )          = mean(log10(pow_spectrogram(:,find(spect_t_speed>=speed_stack(j) & spect_t_speed<=speed_stack(j+1)))),2);
        acceleration_matrix(j,: )   = mean(log10(pow_spectrogram(:,find(spect_t_acceleration>=acceleration_stack(j) & spect_t_acceleration<=acceleration_stack(j+1)))),2);
    end

    figure
    subplot(1,2,1)
    imagesc(f , speed_stack, speed_matrix)
    axis xy
    clim([4.2 5.2])


    subplot(1,2,2)
    imagesc(f , acceleration_stack, acceleration_matrix)
    axis xy


    figure('units','normalized','outerposition',[0 0 .5 1]);
    play_bout_spect_time        = any(spect_time>=play_bouts_table(:,1) & spect_time<=play_bouts_table(:,2));

    subplot(1,2,1)
    matrx2plot = log10(pow_spectrogram(:,play_bout_spect_time==0))';
    mean2plot = mean(matrx2plot);
    [~, ~, ci] = ttest(matrx2plot);
    fill([f fliplr(f)],[ci(1,:) fliplr(ci(2,:))],'k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    hold on
    plot(f, mean2plot, 'k')
    title(ch_n)



    matrx2plot = log10(pow_spectrogram(:,play_bout_spect_time==1))';
    mean2plot = mean(matrx2plot);
    [~, ~, ci] = ttest(matrx2plot);
    fill([f fliplr(f)],[ci(1,:) fliplr(ci(2,:))],'r', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    hold on
    plot(f, mean2plot, 'r')

    xlabel('Freq (Khz)')
    ylabel('Pow')
    legend({'No Play','Play'})

    subplot(4,2,2)

    plot(animal_speed(theta_pow_in_range(1:end-1)),theta_pow(theta_pow_in_range(1:end-1)), '.k' )
    xlabel('Speed (a.u.)')
    ylabel('Pow')

    subplot(4,2,4)
    plot(animal_acceleration(theta_pow_in_range(1:end-2)),theta_pow(theta_pow_in_range(1:end-2)) , '.k' )
    xlabel('Acceleration (a.u.)')
    ylabel('Pow')

    speed_model = fitlm(animal_speed,theta_pow(1:end-1));
    D_speed  = speed_model.SSE;

    acceleration_model = fitlm(animal_acceleration,theta_pow(1:end-2));
    D_acceleration  = acceleration_model.SSE;




    speed_stack             = linspace(floor(min(spect_t_speed)), ceil(max(spect_t_speed)), n_stacks);
    acceleration_stack      = linspace(floor(min(spect_t_acceleration)), ceil(max(spect_t_acceleration)), n_stacks);

    speed_play_matrix            = nan(n_stacks,6);
    acceleration_play_matrix     = nan(n_stacks,6);

    for j=1:n_stacks-1
        data_play   = mean(log10(pow_spectrogram(f_index,find(spect_t_speed>=speed_stack(j) & spect_t_speed<=speed_stack(j+1) & play_bout_spect_time==1 & signal_in_range))));
        data_noplay = mean(log10(pow_spectrogram(f_index,find(spect_t_speed>=speed_stack(j) & spect_t_speed<=speed_stack(j+1) & play_bout_spect_time==0 & signal_in_range))));

        if numel(data_play)>100
            data_play = randsample(  data_play,100);
        end
        if numel(data_noplay)>100
            data_noplay = randsample(data_noplay,100);
        end
        speed_play_matrix(j,1 )  = mean(data_play);
        speed_play_matrix(j,2)   = mean(data_noplay);
        speed_play_matrix(j,3 )  = std(data_play);
        speed_play_matrix(j,4)   = std(data_noplay);
        speed_play_matrix(j,5 )  = numel(data_play);
        speed_play_matrix(j,6)   = numel(data_noplay);

        data_play   = mean(log10(pow_spectrogram(f_index,find(spect_t_acceleration>=acceleration_stack(j) & spect_t_acceleration<=acceleration_stack(j+1) & play_bout_spect_time==1 & signal_in_range))));
        data_noplay = mean(log10(pow_spectrogram(f_index,find(spect_t_acceleration>=acceleration_stack(j) & spect_t_acceleration<=acceleration_stack(j+1) & play_bout_spect_time==0 & signal_in_range))));
        if numel(data_play)>100
            data_play = randsample(  data_play,100);
        end
        if numel(data_noplay)>100
            data_noplay = randsample(data_noplay,100);
        end
        acceleration_play_matrix(j,1 )  = mean(data_play);
        acceleration_play_matrix(j,2)   = mean(data_noplay);
        acceleration_play_matrix(j,3 )  = std(data_play);
        acceleration_play_matrix(j,4)   = std(data_noplay);
        acceleration_play_matrix(j,5 )  = numel(data_play);
        acceleration_play_matrix(j,6)   = numel(data_noplay);

    end


    if ch_n==1
        stack_lm_structure.speed_stack = speed_stack;
        stack_lm_structure.acceleration_stack = acceleration_stack;
        stack_lm_structure.speed_play_matrix = speed_play_matrix;
        stack_lm_structure.acceleration_play_matrix = acceleration_play_matrix;
    else

        stack_lm_structure(ch_n).speed_stack = speed_stack;
        stack_lm_structure(ch_n).acceleration_stack = acceleration_stack;
        stack_lm_structure(ch_n).speed_play_matrix = speed_play_matrix;
        stack_lm_structure(ch_n).acceleration_play_matrix = acceleration_play_matrix;
    end

    subplot(4,2,6)
    ci_low  = (speed_play_matrix(:,1)-speed_play_matrix(:,3)./sqrt(speed_play_matrix(:,5)))';
    ci_high = (speed_play_matrix(:,1)+speed_play_matrix(:,3)./sqrt(speed_play_matrix(:,5)))';
    no_nan  = ~isnan(ci_high + ci_low);
    fill([speed_stack(no_nan) fliplr(speed_stack(no_nan))],[ci_low(no_nan) fliplr(ci_high(no_nan))],'r', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    hold on
    plot(speed_stack,speed_play_matrix(:,1), 'r')

    ci_low  = (speed_play_matrix(:,2)-speed_play_matrix(:,4)./sqrt(speed_play_matrix(:,6)))';
    ci_high = (speed_play_matrix(:,2)+speed_play_matrix(:,4)./sqrt(speed_play_matrix(:,6)))';
    no_nan  = ~isnan(ci_high + ci_low);
    fill([speed_stack(no_nan) fliplr(speed_stack(no_nan))],[ci_low(no_nan) fliplr(ci_high(no_nan))],'k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    hold on
    plot(speed_stack,speed_play_matrix(:,2)', 'k')
    xlabel('Speed (a.u.)')
    ylabel('Pow')
    legend({'Play','No Play'})
    legend('Box', 'off')

    subplot(4,2,8)


    ci_low  = (acceleration_play_matrix(:,1)-acceleration_play_matrix(:,3)./sqrt(acceleration_play_matrix(:,5)))';
    ci_high = (acceleration_play_matrix(:,1)+acceleration_play_matrix(:,3)./sqrt(acceleration_play_matrix(:,5)))';
    no_nan  = ~isnan(ci_high + ci_low);
    fill([acceleration_stack(no_nan) fliplr(acceleration_stack(no_nan))],[ci_low(no_nan) fliplr(ci_high(no_nan))],'r', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    hold on
    plot(acceleration_stack,acceleration_play_matrix(:,1), 'r')

    ci_low  = (acceleration_play_matrix(:,2)-acceleration_play_matrix(:,4)./sqrt(acceleration_play_matrix(:,6)))';
    ci_high = (acceleration_play_matrix(:,2)+acceleration_play_matrix(:,4)./sqrt(acceleration_play_matrix(:,6)))';
    no_nan  = ~isnan(ci_high + ci_low);
    fill([acceleration_stack(no_nan) fliplr(acceleration_stack(no_nan))],[ci_low(no_nan) fliplr(ci_high(no_nan))],'k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    hold on
    plot(acceleration_stack,acceleration_play_matrix(:,2)', 'k')

    xlabel('Acceleration (a.u.)')
    ylabel('Pow')
    legend({'Play','No Play'})
    legend('Box', 'off')


    figure
    lm_table= array2table([spect_t_speed' spect_t_acceleration' play_bout_spect_time' theta_pow_spect_t']);
    lm_table.Properties.VariableNames = {'Speed','Acceleration','Play','ThetaPow'};
    lm_table = lm_table(signal_in_range,:);
    lm_table.Play = categorical(  lm_table.Play);

    lm_speed_play = fitlm(lm_table, 'ThetaPow~Play*Acceleration + Play*Speed + Play');
    figure()
    % gscatter(lm_table.Speed,lm_table.ThetaPow,lm_table.Play,'br','.o')
    hold on
    line_plot =unique([lm_table.Speed(lm_table.Play=="1"),feval(lm_speed_play,lm_table(lm_table.Play=="1",:))], 'rows');
    line(line_plot(:,1), line_plot(:,2),'Color','r','LineWidth',4)
    line_plot =unique([lm_table.Speed(lm_table.Play=="0"),feval(lm_speed_play,lm_table(lm_table.Play=="0",:))], 'rows');
    line(line_plot(:,1), line_plot(:,2),'Color','b','LineWidth',4)


    if ch_n==1
        stack_lm_structure.lm_table = lm_table;
        stack_lm_structure.lm_speed_play = lm_speed_play;

    else

        stack_lm_structure(ch_n).lm_table = lm_table;
        stack_lm_structure(ch_n).lm_speed_play = lm_speed_play;

    end


end

% %%
% figure 
% hmm_states_spect_time       = any(spect_time>=beg_end_times(:,1) & spect_time<=beg_end_times(:,2));
% 
% 
% 
% 
% 
% 
% 
% subplot(1,2,1)
% matrx2plot = log10(pow_spectrogram(:,hmm_states_spect_time==0))';
% mean2plot = mean(matrx2plot);
% [~, ~, ci] = ttest(matrx2plot);
% fill([f fliplr(f)],[ci(1,:) fliplr(ci(2,:))],'k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
% hold on
% plot(f, mean2plot, 'k')
% 
% 
% 
% matrx2plot = log10(pow_spectrogram(:,hmm_states_spect_time==1))';
% mean2plot = mean(matrx2plot);
% [~, ~, ci] = ttest(matrx2plot);
% fill([f fliplr(f)],[ci(1,:) fliplr(ci(2,:))],'r', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
% hold on
% plot(f, mean2plot, 'r')
% 
% subplot(4,2,2)
% 
% plot(animal_speed,theta_pow(1:end-1), '.k' )
% subplot(4,2,4)
% 
% plot(animal_acceleration,theta_pow(1:end-2) , '.k' )
% 
% 
% speed_model = fitlm(animal_speed,theta_pow(1:end-1));
% D_speed  = speed_model.SSE;
% 
% acceleration_model = fitlm(animal_acceleration,theta_pow(1:end-2));
% D_acceleration  = acceleration_model.SSE;
% 
% 
% 
% 
% speed_stack             = linspace(floor(min(spect_t_speed)), ceil(max(spect_t_speed)), n_stacks);
% acceleration_stack      = linspace(floor(min(spect_t_acceleration)), ceil(max(spect_t_acceleration)), n_stacks);
% 
% speed_play_matrix            = nan(n_stacks,2);
% acceleration_play_matrix     = nan(n_stacks,2);
% 
% for j=1:n_stacks-1
% speed_play_matrix(j,1 )          = mean(mean(log10(pow_spectrogram(f_index,find(spect_t_speed>=speed_stack(j) & spect_t_speed<=speed_stack(j+1) & hmm_states_spect_time==1 & signal_in_range))),2));
% speed_play_matrix(j,2)          = mean(mean(log10(pow_spectrogram(f_index,find(spect_t_speed>=speed_stack(j) & spect_t_speed<=speed_stack(j+1) & hmm_states_spect_time==0 & signal_in_range))),2));
% 
% acceleration_play_matrix(j,1 )   = mean(mean(log10(pow_spectrogram(f_index,find(spect_t_acceleration>=acceleration_stack(j) & spect_t_acceleration<=acceleration_stack(j+1) & hmm_states_spect_time==1 & signal_in_range))),2));
% acceleration_play_matrix(j,2 )   = mean(mean(log10(pow_spectrogram(f_index,find(spect_t_acceleration>=acceleration_stack(j) & spect_t_acceleration<=acceleration_stack(j+1) & hmm_states_spect_time==0 & signal_in_range))),2));
% 
% end
% 
% 
% subplot(4,2,6)
% plot(speed_stack,speed_play_matrix(:,1), 'r')
% hold on
% plot(speed_stack,speed_play_matrix(:,2)', 'k')
% 
% 
% subplot(4,2,8)
% 
% plot(acceleration_stack,acceleration_play_matrix(:,1), 'r')
% hold on
% plot(acceleration_stack,acceleration_play_matrix(:,2)', 'k')