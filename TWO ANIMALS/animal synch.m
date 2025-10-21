
load( 'synch_model_video2NPX.mat')

% load('traking_structure.mat')
Behavior_file = dir('ELAN*');
Behavior_file =Behavior_file.name;
animal_info =cd;
animal_info = strsplit(animal_info, '\');
animal_info = animal_info{end};
animal_info = strsplit(animal_info, ' ');
% Behavior_file ='ELAN behavior tabulated.txt'
% Behavior_file ='_NMMTT_230929 23-11-05 16-48-09';
% Behavior_file = '_NMMTT_230929 23-10-20 13-55-51.txt';
 % Behavior_file= 'C2+C1_SP_2006_0003 25-06-20 15-48-15';
% Behavior_file ='C1+PD1_SP_2306_0002 25-06-23 16-40-50';
play_behaviors      = {'Pounce', 'CC','Boxing', 'Evasion','Pin', 'Escape', 'CB', 'CD'};

%% 2 Load Behavior

animal_1     = animal_info{1};

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
partner_names(ismember(animal_types, animal_1))                  = [];

config.Behavior         = Behavior;
config.repeated_animal  = animal_1;
config.animal_types     = animal_types        ;
config.play_behaviors   = play_behaviors      ;
config.beh_bin          = bin_size             ;
config.conv_length      = conv_length;
config.behavior_window  = 0;


[play_bouts_table]      = play_bout(config);


Unresponsiveness = find(ismember(Behavior.Type, 'Unresponsive'));

play_session = Behavior{ismember(Behavior.Type, 'Partners session'), {'Start', 'End'}};
if isempty(play_session)
    play_session = [min(Behavior.Start)-5 max(Behavior.End)+5]
end
UnResp  = zeros(size(play_bouts_table,1),2);

for j=Unresponsiveness'

    indexes =   (play_bouts_table(:,1)>=Behavior.Start(j) & play_bouts_table(:,1)<= Behavior.End(j)) | ...
        (play_bouts_table(:,2)>=Behavior.Start(j) & play_bouts_table(:,2)<= Behavior.End(j));

    animal_2 = Behavior.Animal(j);
    UnResp(indexes,:) = 1;
    UnResp(indexes,2) = find(ismember(animal_types,animal_2));
end








%%  load lfp from current dir


channel_list = readtable('channel selection.txt');

disp('LOADING animal 1')
animal_1_ch = channel_list.Var2(ismember(channel_list.Var1, animal_info{1}));
file_pointer                = fopen(['continuous_' ,animal_info{1},'.dat'], 'r');
% file_pointer                = fopen('continuous_SINGLE2.dat', 'r');
LFP_animal_1_PAG             = fread(file_pointer,'int16');
LFP_animal_1_PAG             = reshape(LFP_animal_1_PAG, 384, numel(LFP_animal_1_PAG)/384);
lfp_animal_1 = double(LFP_animal_1_PAG(animal_1_ch,:));
clear LFP_animal_1_PAG

disp('LOADING animal 2')
file_pointer                = fopen(['continuous_' ,animal_info{2},'.dat'], 'r');
% file_pointer                = fopen('continuous_SINGLE4.dat', 'r');
LFP_animal_2_PAG            = fread(file_pointer,'int16');
LFP_animal_2_PAG             = reshape(LFP_animal_2_PAG, 384, numel(LFP_animal_2_PAG)/384);
animal_2_ch = channel_list.Var2(ismember(channel_list.Var1, animal_info{2}));
lfp_animal_2 = double(LFP_animal_2_PAG(animal_2_ch,:));
clear LFP_animal_2_PAG

play_song([],[],[])


%% load lfp for npx2



%%
area_limit_table    = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\Area_limits_GoodLooking.xlsx';
% for Single2 and Single4
animal_1        = 'Batch2Single2';
animal_2        = 'Batch2Single4';

% for Dual and Single
% animal_1        = 'Batch1Single3';
% animal_2        = 'Batch1Dual1';


area_limit = readtable(area_limit_table);
area_limit = area_limit(ismember(area_limit.AnimalName,animal_1),:);
PAG_channels_animal_1 = area_limit{ismember(area_limit.area, {'LPAG'}), {'ch_start', 'ch_end'}};
PAG_channels_animal_1 = str2double(PAG_channels_animal_1);
channel_Range = [min(PAG_channels_animal_1(:)) max(PAG_channels_animal_1(:))];
mid_PAG_channel_animal1 = round(mean(channel_Range));
lfp_animal_1 = double(LFP_animal_1_PAG(mid_PAG_channel_animal1,:));
clear LFP_animal_1_PAG

area_limit = readtable(area_limit_table);
area_limit = area_limit(ismember(area_limit.AnimalName,animal_2),:);
PAG_channels_animal_2 = area_limit{ismember(area_limit.area, {'LPAG'}), {'ch_start', 'ch_end'}};
PAG_channels_animal_2 = str2double(PAG_channels_animal_2);
channel_Range = [min(PAG_channels_animal_2(:)) max(PAG_channels_animal_2(:))];
mid_PAG_channel_animal2= round(mean(channel_Range));
lfp_animal_2 = double(LFP_animal_2_PAG(mid_PAG_channel_animal2,:));
clear LFP_animal_2_PAG
%% olny for NPX2 (natalies datasets)

mid_PAG_channel_animal1 = 76; % #t6 for C1
lfp_animal_1 = double(LFP_animal_1_PAG(mid_PAG_channel_animal1,:));
clear LFP_animal_1_PAG
mid_PAG_channel_animal2 = 11; % #11 for C2, #347 for PD1
lfp_animal_2 = double(LFP_animal_2_PAG(mid_PAG_channel_animal2,:));
clear LFP_animal_2_PAG





%% same as before but now using power spectrum isntead of hilbert transform
wind_length     = 1;
wind_overlap    = .990;
f               = .1:.1:6;
freq_pow_range  = [1 4];

% wind_length     = .250;
% wind_overlap    = .240;
% f               = 5:.1:14;
% freq_pow_range  = [6 12];
sr = 2500;
index_play_session_range = round((play_session(1)-5)*sr:(play_session(2)+5)*sr);
time_play_session = index_play_session_range/sr;

  [pow_spectrogram,~,spect_time]  = spectrogram(lfp_animal_1(index_play_session_range),wind_length*2500, wind_overlap*2500, f,2500);
    spect_time                     = spect_time+play_session(1)-5;
    pow_spectrogram = abs(pow_spectrogram);
    mean_pow_animal_1 = mean(pow_spectrogram(f >=freq_pow_range(1) & f<=freq_pow_range(2),:));


     [pow_spectrogram,~,~]  = spectrogram(lfp_animal_2(index_play_session_range),wind_length*2500, wind_overlap*2500, f,2500);
    pow_spectrogram = abs(pow_spectrogram);
    mean_pow_animal_2 = mean(pow_spectrogram(f >=freq_pow_range(1) & f<=freq_pow_range(2),:));
    play_song([],[],[])


 %%
 spect_sr = round(1/(wind_length-wind_overlap));
time_range = [-10 10];
psth_time =( time_range(1)*spect_sr:time_range(2)*spect_sr)/spect_sr;
baseline = psth_time<-1;
n_samples = round((range(time_range)*spect_sr) + 1);
psth_onset = nan(2,size(play_bouts_table, 1),n_samples);
psth_offset = nan(2,size(play_bouts_table, 1),n_samples);

psth_zscored = nan(2,size(play_bouts_table, 1),n_samples);
trial_correlation =nan(size(play_bouts_table, 1),1);
for j=1:size(play_bouts_table, 1)

    beg_time = play_bouts_table(j,1);
    pb_length = play_bouts_table(j,2)-beg_time;

    [~, loc] = min(abs(spect_time-beg_time));

    index2exctact = round((loc+time_range(1)*spect_sr):(loc+time_range(2)*spect_sr));
    possible_index = ismember(index2exctact,1:numel(spect_time));
    psth_onset(1,j,possible_index) = (mean_pow_animal_1(index2exctact(possible_index)));
    psth_onset(2,j,possible_index) = (mean_pow_animal_2(index2exctact(possible_index)));

    psth_zscored(1,j,:)  = ( psth_onset(1,j,:) - mean( psth_onset(1,j,baseline), 'omitmissing'))/ std( psth_onset(1,j,baseline), 'omitmissing');
    % psth_zscored(1,j,psth_time>length) = NaN;
    psth_zscored(2,j,:)  = ( psth_onset(2,j,:) - mean( psth_onset(2,j,baseline), 'omitmissing'))/ std( psth_onset(2,j,baseline), 'omitmissing');
    % psth_zscored(2,j,psth_time>length) = NaN;

    trial_correlation(j) = corr(squeeze(psth_zscored(1,j,psth_time<pb_length & psth_time>0) ), squeeze(psth_zscored(2,j,psth_time<pb_length  & psth_time>0)));

    end_time = play_bouts_table(j,2);


    [~, loc] = min(abs(spect_time-end_time));

    index2exctact = round((loc+time_range(1)*spect_sr):(loc+time_range(2)*spect_sr));
    possible_index = ismember(index2exctact,1:numel(spect_time));
    psth_offset(1,j,possible_index) = (mean_pow_animal_1(index2exctact(possible_index)));
    psth_offset(2,j,possible_index) = (mean_pow_animal_2(index2exctact(possible_index)));

end

a = psth_onset(1,:,:);
a = a(:);
b =  psth_onset(2,:,:);
b = b(:);

figure
plot(a, b, '.k')
no_nan = ~isnan(a+b);
[c,p] = corr(a(no_nan),b(no_nan));
title([c,p])


%%
 [~, order] = sort(diff(play_bouts_table'));
 only_responsive = UnResp(:,1)==0;
 % non_outliers_1 = max(abs(zscore(squeeze(psth_zscored(1,order,:)))), [],2)<3.5;
 % non_outliers_2 = max(abs(zscore(squeeze(psth_zscored(2,order,:)))), [],2)<3.5;
 % non_outliers  = non_outliers_1 & non_outliers_2;
 non_outliers = true;
 order = order(only_responsive & non_outliers);
figure
subplot(3,1,1)
imagesc(psth_time,1:size(play_bouts_table(only_responsive & non_outliers), 1),squeeze(psth_zscored(1,order,:)))
axis xy
clim([-2 2])
xlim([-1 4])
subplot(3,1,2)
imagesc(psth_time,1:size(play_bouts_table(only_responsive & non_outliers), 1),squeeze(psth_zscored(2,order,:)))
clim([-2 2])
xlim([-1 4])
axis xy
subplot(3,1,3)

I1 = squeeze(psth_zscored(1,:,:));
I2 = squeeze(psth_zscored(2,:,:));

mi = image_mutual_info(I1, I2, 50);
nIter = 1000;  % number of randomizations
MI_rand = zeros(1,nIter);


for k = 1:nIter
    rand_shift = randperm(size(I2,2), size(I2,1));
    % I2_shuffled = circshift(I2,rand_shift); 
    I2_shuffled = I2(randperm(size(I2,1)),:);
    MI_rand(k) = image_mutual_info(I1,   I2_shuffled, 50);
end



histogram(MI_rand, 100, 'EdgeColor','none', 'FaceColor','k')
hold on
y_lim = ylim;
plot([mi mi], y_lim, 'r')
title(['p= ', num2str(sum(MI_rand>=mi)/nIter), ' mi=', num2str(mi)])
%%

t_size = 200;
nIter = 10000;
MI_rand = zeros(1,nIter);


for k = 1:nIter
    rand_shift = randperm(size(I2,2), size(I2,1));
    % I2_shuffled = circshift(I2,rand_shift); 
    I2_shuffled = I2(randperm(size(I2,1)),1:t_size);
    MI_rand(k) = image_mutual_info(I1(:,1:t_size),   I2_shuffled, 10);
end
% generate_rand_distribution
%%

% I1 = squeeze(psth_onset(1,:,:));
% I2 = squeeze(psth_onset(2,:,:));

[n,T] = size(I1);

mi_t = nan(T-t_size,1);
mi_t_pctl =mi_t;
mi_t_by_trial = nan(n,T-t_size,1);
for t=1:T-t_size

    index = t:(t+t_size-1);
    mi_t(t) = image_mutual_info(I1(:,index), I2(:,index), 10);
    mi_t_pctl(t) = sum(MI_rand>mi_t(t))/nIter;

    for trial=1:n
        if ~(all(isnan(I1(trial,index))) ||  all(isnan(I2(trial,index))))
            mi_t_by_trial(trial,t) =  image_mutual_info(I1(trial,index), I2(trial,index), 10);
        end
    end
end

%%
pb_lengths = diff(play_bouts_table')';
 [pb_length_sorted, order] = sort(pb_lengths);

mi_time = psth_time(1:T-t_size) + .5*t_size/spect_sr;
baseline_index =mi_time<0;
mi_t_by_trial_bc = mi_t_by_trial;
mean_mi = nan(n,1);
max_mi = nan(n,1);
before_after_mi = nan(n,2);
start_index = mi_time>-1 & mi_time<0;

for  trial=1:n
        this_mi =   mi_t_by_trial(trial,:) ;


        mean_mi(trial) = mean(mi_t_by_trial(trial,mi_time>0 & mi_time<pb_lengths(trial)));
         max_mi(trial) = max(mi_t_by_trial(trial,mi_time>0 & mi_time<pb_lengths(trial)));
        mi_t_by_trial_bc(trial,:) = (this_mi -mean(this_mi(baseline_index)))/std(this_mi(baseline_index));

        before_after_mi(trial,1) = mean(mi_t_by_trial(trial,start_index))-mean_mi(trial);
        end_index   = mi_time-pb_lengths(trial)>0 & mi_time-pb_lengths(trial)<1;
        before_after_mi(trial,2) = mean(mi_t_by_trial(trial,end_index), 'omitmissing')-mean_mi(trial);


end
figure

subplot(5,1,1:2)

    imagesc(psth_time(1:T-t_size) + .5*t_size/spect_sr,1:n,mi_t_by_trial(order,:))
hold on
plot(pb_length_sorted*0, 1:numel(pb_length_sorted), 'w')
plot(pb_length_sorted, 1:numel(pb_length_sorted) ,'w')

axis xy
subplot(5,1,3)
plot(mi_time,(1-mi_t_pctl))

subplot(5,1,4:5)
plot(mi_time,mi_t)
hold on
plot(mi_time,mean(mi_t_by_trial_bc, 'omitmissing'))

%%

figure
subplot(1,2,1)
plot(mean_mi,pb_lengths, '.')

subplot(1,2,2)
plot(max_mi,pb_lengths, '.')

%%

figure
plot([1 2],before_after_mi, ':k')
hold on
plot([1 2],mean(before_after_mi, 'omitmissing'), 'k')
%%
mi_structure = [];
mi_structure.mi_t_pctl          = mi_t_pctl;
mi_structure.mi_t_pctl          = mi_t;
mi_structure.mi_t_by_trial_bc   = mi_t_by_trial_bc;
mi_structure.mean_mi            = mean_mi;
mi_structure.max_mi             = max_mi;
mi_structure.psth_time          = psth_time;
mi_structure.psth_zscored       = psth_zscored;
mi_structure.psth               = psth_onset;
mi_structure.psth               = psth_offset;
mi_structure.play_bouts_table   = play_bouts_table;
mi_structure.t_size             = t_size;
mi_structure.spect_sr           = spect_sr;



save('mi_structure_C1PD122500620','mi_structure')




%%
%% From here on, only if there is also traking
%% From here on, only if there is also traking
%% From here on, only if there is also traking
%%

traking_time        = predict(synch_model_video2NPX, (traking_structure.frames2stract/30)')';
full_traking_time   = (traking_structure.frames2stract(1):traking_structure.frames2stract(end))/30;
full_traking_time   = predict(synch_model_video2NPX, full_traking_time')';
animal_pos          = nan(numel(full_traking_time),2);
partner_pos         = nan(numel(full_traking_time),2);
animal_pos(:,1)     = interp1(traking_time, smoothdata(traking_structure.animal_pos(:,1), 'loess',5), full_traking_time,'cubic');
animal_pos(:,2)     = interp1(traking_time, smoothdata(traking_structure.animal_pos(:,2), 'loess',5), full_traking_time,'cubic');
partner_pos(:,2)     = interp1(traking_time,smoothdata(traking_structure.partner_pos(:,2), 'loess',5), full_traking_time,'cubic');
partner_pos(:,1)     = interp1(traking_time,smoothdata(traking_structure.partner_pos(:,1), 'loess',5), full_traking_time,'cubic');

animal_dist = sqrt(sum((animal_pos-partner_pos).*(animal_pos-partner_pos),2));

animal_dist_lfp_time = interp1(full_traking_time,animal_dist,lfp_time);

%%

distance_index = (animal_dist_lfp_time<Inf)';
lfp_time = (1/sr):(1/sr):(numel(amplitud_animal_1)/sr);
play_indexes = any(lfp_time>=play_bouts_table(:,1) & lfp_time<=play_bouts_table(:,2))';
limit_2_central_values = abs(zscore(amplitud_animal_1))<1 & abs(zscore(amplitd_animal_2))<1;
no_play_index   = distance_index & limit_2_central_values & ~play_indexes;
play_index      = distance_index & limit_2_central_values & play_indexes;
[r,lags] = xcorr(amplitud_animal_1(no_play_index),amplitd_animal_2(no_play_index), sr*10, 'coeff');
[r_play,lags_play] = xcorr(amplitud_animal_1(play_index),amplitd_animal_2(play_index), sr*10, 'coeff');
figure
plot(lags/sr, r, 'k')
hold on
plot(lags_play/sr, r_play, 'r')

%%

play_indexes = any(lfp_time>=play_bouts_table(:,1) & lfp_time<=play_bouts_table(:,2))';
limit_2_central_values = abs(zscore(amplitud_animal_1))<1 & abs(zscore(amplitd_animal_2))<1;

distance_var = linspace(min(animal_dist_lfp_time), max(animal_dist_lfp_time), 25);
r_by_distP_noplay = nan(numel(distance_var)-1,2*sr*10  +1);
r_by_distP_play = nan(numel(distance_var)-1,2*sr*10  +1);


for n=1:numel(distance_var)-1

    distance_index = (animal_dist_lfp_time>distance_var(n) & animal_dist_lfp_time<=distance_var(n+1))';
    lfp_time = (1/sr):(1/sr):(numel(amplitud_animal_1)/sr);

    no_play_index   = distance_index & limit_2_central_values & ~play_indexes;
    play_index      = distance_index & limit_2_central_values & play_indexes;
    [r,lags] = xcorr(amplitud_animal_1(no_play_index),amplitd_animal_2(no_play_index), sr*10, 'coeff');
    r_by_distP_noplay(n,:) = r;
    [r,lags] = xcorr(amplitud_animal_1(play_index),amplitd_animal_2(play_index), sr*10, 'coeff');
    r_by_distP_play(n,:) = r;
end
% [r_play,lags_play] = xcorr(amplitud_single(play_index),amplitd_double(play_index), sr*10, 'coeff');
% figure
% plot(lags/sr, r, 'k')
% hold on
% plot(lags_play/sr, r_play, 'r')
figure
subplot(1,2,1)
imagesc(lags/sr,distance_var, r_by_distP_noplay)
clim([0 1])
axis xy

subplot(1,2,2)
imagesc(lags/sr,distance_var, r_by_distP_play)
clim([0 1])
axis xy

%%

n2plot = 2500;
n=4;
play_indexes = any(lfp_time>=play_bouts_table(:,1) & lfp_time<=play_bouts_table(:,2))';
limit_2_central_values = abs(zscore(amplitud_animal_1))<1.5 & abs(zscore(amplitd_animal_2))<1.5;

distance_index = (animal_dist_lfp_time>distance_var(n))';
lfp_time = (1/sr):(1/sr):(numel(amplitud_animal_1)/sr);

no_play_index   = find(distance_index & limit_2_central_values & ~play_indexes);
play_index      = find(distance_index & limit_2_central_values & play_indexes);

play2plot = randsample(play_index,n2plot);
no_play2plot = randsample(no_play_index,n2plot);
figure
subplot(1,2,1)
plot(amplitud_animal_1(no_play2plot),amplitd_animal_2(no_play2plot), '.')
hold on
plot(amplitud_animal_1(play2plot),amplitd_animal_2(play2plot), '.')

[c,p] = corr(amplitud_animal_1(play2plot),amplitd_animal_2(play2plot))
[c,p] = corr(amplitud_animal_1(no_play2plot),amplitd_animal_2(no_play2plot))