%%  define bfolders
figure_dir      = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Codes\Figure codes\Figure 1 Inputs';
synch_folder    = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Synch data';
hmm_raw_data    = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\HMM data\HMM raw data';
call_folder     = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\CallDetectionBackup';
behavior_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Behavior backups';
analyssis_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\HMM 2 and 3 states 2 partners';
animal_name     = 'B1D1 1013 Dual';
repeated_animal = strsplit(animal_name, ' ');
repeated_animal = repeated_animal{end};
beh_bin = 0.01; %for hmm stimate
start_event = 480;
end_event   = 490;

load([synch_folder, '\',animal_name, '\synch_model_video2audio.mat'], 'synch_model_video2audio')

Behavior            =   readtable([behavior_folder,'\',animal_name, '.txt']); 
Behavior(:,2)       = [];
Behavior.Properties.VariableNames = {'Animal', 'Start', 'End', 'Length', 'Type'};
Behavior.Start      = predict(synch_model_video2audio, Behavior.Start);
Behavior.End        = predict(synch_model_video2audio, Behavior.End);
Behavior.Type2      = Behavior.Type;
Behavior.Type2(ismember(Behavior.Type2, {'Pounce_A', 'Pounce_B'})) = {'Pounce'}; %% Merging behaviors to Type2
Behavior.Type2(ismember(Behavior.Type2, {'Pounce_Ai', 'Pounce_Bi'})) = {'PounceI'};
Behavior.Type2(ismember( Behavior.Type2,'')) = {'Other'};
Behavior(ismember(Behavior.Animal, 'Reversal'),:) = [];


min_time2analysis = Behavior.Start(ismember(Behavior.Type,'Partners session'))  ;
max_time2analysis = Behavior.End(ismember(Behavior.Type,'Partners session'))  ;
ListOfPartners = [min_time2analysis max_time2analysis];
[~, time_ordered] = sort(ListOfPartners(:,1));
ListOfPartners = ListOfPartners(time_ordered,:);

pt = find(ListOfPartners(:,1)<=start_event & ListOfPartners(:,2)>=start_event);

load([hmm_raw_data,'\',animal_name, ' P', num2str(pt), '_PropAndTime'], 'adjusted_binned_time', 'all_properties', 'ALL_VARIABLE_NAMES')

variable_name2states        =  [hmm_raw_data, '\',animal_name, ' P', num2str(pt),'_states_K2.npy'];
% variable_behavior_prop      =  [hmm_raw_data, '\',animal_name, ' P', num2str(pt),'.npy'];
time_limit = (adjusted_binned_time>= ListOfPartners(pt,1) & adjusted_binned_time<=  ListOfPartners(pt,2) );
% time_limit = (adjusted_binned_time>= min(min(play_bouts_table)) & adjusted_binned_time<=   max(max(play_bouts_table)) );

hmm_states      =  readNPY(variable_name2states);
% behavior_prop   = readNPY(variable_behavior_prop);


sr          = 250000;

CallStats   = readtable([call_folder, '\',animal_name,'_Stats.xlsx']);
CallStats.Properties.VariableNames = cellfun(@(x) strrep(x, '_', ''),CallStats.Properties.VariableNames, 'UniformOutput',false );

[audio_data, sr] =  audioread([call_folder, '\', animal_name, '.wav'], round([start_event*sr  end_event*sr ]));



time_index = adjusted_binned_time>=start_event & adjusted_binned_time<=end_event;

%% estiamte call spectrogram (take some time)
fs = 250000;
f = 20000:100000;
window_insec        = 0.02;
overlap_insec       = 0.018;

window              = round(window_insec*fs);
noverlap            = round(overlap_insec*fs);

[s,f,t] = spectrogram(audio_data,window,noverlap,f,fs);
play_song([],[],[])

%% smooth image (also take some time)

pow = log10(abs(s));
% Parameters
sigma = std(pow(:)) ;       % Standard deviation
kernel_size = ceil(2*6*sigma);  % Kernel size (usually ~6*sigma for full support)

% Create grid
[x, y] = meshgrid(-kernel_size:kernel_size, -kernel_size:kernel_size);
gaussian_kernel = exp(-(x.^2 + y.^2) / (2*sigma^2));
gaussian_kernel = gaussian_kernel / sum(gaussian_kernel(:));  % Normalize

smoothed_data = conv2(pow, gaussian_kernel, 'same');
I_sharp = imsharpen(smoothed_data);

h = fspecial('unsharp');         % Unsharp mask (a high-pass filter)
I_highcontrast = imfilter(smoothed_data, h);
%% add call subplot to figure (PANEL B)
behaviors_in_range = find((Behavior.Start>=start_event & Behavior.Start<=end_event) | (Behavior.End>=start_event & Behavior.End<=end_event))';
figure
colormap(1-gray)
subplot(4,1,1)

mean_call_length = median(CallStats.CallLengths);

imagesc(t+start_event, f, I_sharp)
axis xy
clim([-3.5 -.5])
ylim([35 80]*1000)
xlim([start_event end_event])
set(gca, 'TickDir', 'out')
xticks([start_event start_event+1])
xticklabels([])

%% call rate, and other variables (PANEL B)
mov_win_sec = .5;
mov_win = mov_win_sec/beh_bin;
subplot(4,1,2)
hold off
var_2_plot      = {'NumCalls'};
var2_plot_index = find(ismember(ALL_VARIABLE_NAMES,var_2_plot));
plot(adjusted_binned_time(time_index), movmean(all_properties(time_index,var2_plot_index(1)),mov_win)/(mean_call_length/mov_win_sec), 'k')
ylabel('Call Rate (Hz)')
xlim([start_event end_event])
set(gca, 'TickDir', 'out')
xticks([start_event start_event+1])
xticklabels([])

var_2_plot      = {'AnimalSpeed' , 'PartnerSpeed'};
var2_plot_index = find(ismember(ALL_VARIABLE_NAMES,var_2_plot));

subplot(4,1,3)
set(gca, 'TickDir', 'out')
hold off

plot(adjusted_binned_time(time_index), all_properties(time_index,var2_plot_index(1)), 'k')
hold on
y_lim1 = ylim;
ylabel('Speed (a.u.)')


subplot(4,1,4)
hold off
plot(adjusted_binned_time(time_index), all_properties(time_index,var2_plot_index(2)), 'k')
hold on
y_lim2 = ylim;
xlim([start_event end_event])
ylabel('Speed (a.u.)')


hold on

for bn=behaviors_in_range
        beh_start   = Behavior.Start(bn);
        beh_end     = Behavior.End(bn);
        animal_type = Behavior.Animal(bn);
        behavior_type = Behavior.Type{bn};

        if ismember(animal_type,repeated_animal)
            subplot(4,1,3)
            fill([beh_start beh_end beh_end beh_start], y_lim1([1 1 2 2]), 'k', 'FaceAlpha',.5, 'EdgeColor','k')
            text((beh_end+beh_start)/2, mean(y_lim1), behavior_type, 'Color', 'r')
            xlim([start_event end_event])
            set(gca, 'TickDir', 'out')
            xticks([start_event start_event+1])
            xticklabels([])
        else
            subplot(4,1,4)
            fill([beh_start beh_end beh_end beh_start], y_lim2([1 1 2 2]), 'k', 'FaceAlpha',.5, 'EdgeColor','k')
            text((beh_end+beh_start)/2, mean(y_lim1), behavior_type,  'Color', 'r')
            xlim([start_event end_event])
            set(gca, 'TickDir', 'out')
            xticks([start_event start_event+1])    

        end
end

%% SAVE PANEL B
print(gcf,'-vector','-dsvg',[figure_dir,'\raw data scheme.svg'])
clear I_sharp pow I_highcontrast smoothed_data
%% now plot pca and hmm on top (For PANEL F)

play_val=0;
load([hmm_raw_data,'\',animal_name,  ' P', num2str(pt), ' binned time.mat'],'adjusted_binned_time');
load([hmm_raw_data,'\',animal_name,  ' P', num2str(pt), '_PropAndTime.mat'],'all_properties', 'ALL_VARIABLE_NAMES','classification_matrix4HMM')
[coef, score] = pca(classification_matrix4HMM);
hmm_states = readNPY([hmm_raw_data,'\',animal_name,  ' P', num2str(pt), '_states_K2.npy']);

binned_time_index = adjusted_binned_time>=start_event & adjusted_binned_time<=end_event;
this_time = adjusted_binned_time(binned_time_index);
this_score = score(binned_time_index, 1);

figure
plot(this_time,this_score, 'k')
d = diff([0 hmm_states(bined_time_index)'==play_val 0]);  % pad with zeros at both ends
startIdx = find(d == 1);   % start of 1-sequences
endIdx = find(d == -1) - 1;
hold on
subtimes = [sub_Time(startIdx)' sub_Time(endIdx)'];
for j=1:size(subtimes,1)
hmm_start   = subtimes(j,1);
hmm_end     = subtimes(j,2);
fill([hmm_start hmm_end hmm_end hmm_start], y_lim([1 1 2 2]), 'c', 'FaceAlpha',.25)
end

%% now make hmm with play beahviors (here start loading hmm data)
 last_hmm_dir = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\HMM 2 and 3 states 2 partners';


prediction_struct_files = dir([last_hmm_dir, '\* prediction_struct*']);
is_there_play_bout      = [];
is_there_hmm            = [];
is_this_hmm             = [];
filled_play_bouts       = [];
tripple_states          = [];
re_assignment           = [];
is_there_play_beh       = [];
session_index           = [];



total_number_of_hmm = 0;
percentage_of_bouts_with_play =nan(numel(prediction_struct_files),1);

for fn= 1:numel(prediction_struct_files)

    figure
     load([last_hmm_dir,'\',prediction_struct_files(fn).name]) 
     psth_edges = prediction_struct.psth_edges;
    [hmm_length_ordered, pb_order] = sort(diff(prediction_struct.filled_play_bouts'));
    imagesc(psth_edges, 1:numel(hmm_length_ordered), 1-cat(3,prediction_struct.is_there_hmm(pb_order,:), prediction_struct.is_there_play_beh(pb_order,:), prediction_struct.is_there_play_beh(pb_order,:) ))
    title(sum(sum(prediction_struct.is_this_hmm & prediction_struct.is_there_play_beh,2)>0))
    percentage_of_bouts_with_play(fn) = sum(sum(prediction_struct.is_this_hmm & prediction_struct.is_there_play_beh,2)>0)/size(prediction_struct.is_this_hmm,1);
    axis xy
    hold on
    plot(hmm_length_ordered,1:numel(hmm_length_ordered),'k')
    plot(hmm_length_ordered*0,1:numel(hmm_length_ordered),'k')
    
     is_there_play_bout      = [is_there_play_bout;prediction_struct.is_there_play_bout];
     is_there_play_beh       = [is_there_play_beh;prediction_struct.is_there_play_beh];
     is_there_hmm            = [is_there_hmm;prediction_struct.is_there_hmm];
     is_this_hmm             = [is_this_hmm;prediction_struct.is_this_hmm];
     filled_play_bouts       = [filled_play_bouts;prediction_struct.filled_play_bouts];
     tripple_states          = [tripple_states;prediction_struct.what_3states_is];
     re_assignment           = [re_assignment;prediction_struct.re_assignment]
     session_index           = [session_index;ones(size(prediction_struct.filled_play_bouts,1),1)*fn];
end
% current_hmm = tripple_states==3;
psth_edges = prediction_struct.psth_edges;

%% Estiamte cumulative engagned and unegnaged lenght distributions (PANEL K)
hmm_length_edges = 0:.1:15;
hmm_length_edges_centers = .5*(hmm_length_edges(2:end) + hmm_length_edges(1:end-1));


distribution_lengths_play         =nan(numel(prediction_struct_files),numel(hmm_length_edges)-1);
distribution_lengths_without_play =nan(numel(prediction_struct_files),numel(hmm_length_edges)-1);

cum_distribution_lengths_play         =nan(numel(prediction_struct_files),numel(hmm_length_edges)-1);
cum_distribution_lengths_without_play =nan(numel(prediction_struct_files),numel(hmm_length_edges)-1);

hmm_lenghts = diff(filled_play_bouts')';

for sn=1:numel(prediction_struct_files)
        

hmm_lengths_with_play       = hmm_lenghts(sum(is_this_hmm & is_there_play_beh,2)>0 & session_index==sn);
hmm_lengths_without_play    = hmm_lenghts(sum(is_this_hmm & is_there_play_beh,2)<=0 & session_index==sn);
n_events = sum(session_index==sn);
distribution_lengths_play(sn,:) = movmean((histcounts(hmm_lengths_with_play,hmm_length_edges))/numel(hmm_lengths_with_play),10);
distribution_lengths_without_play(sn,:) = movmean((histcounts(hmm_lengths_without_play,hmm_length_edges))/numel(hmm_lengths_without_play),10);

cum_distribution_lengths_play(sn,:) = movmean(cumsum(histcounts(hmm_lengths_with_play,hmm_length_edges))/numel(hmm_lengths_with_play),10);
cum_distribution_lengths_without_play(sn,:) = movmean(cumsum(histcounts(hmm_lengths_without_play,hmm_length_edges))/numel(hmm_lengths_without_play),10);


end
%% plot cumulative distribution of lenghths (PANEL K)
line_width = 2.5;
figure
subplot(1,2,1)
plot(hmm_length_edges_centers,distribution_lengths_without_play, ':k')
hold on
plot(hmm_length_edges_centers,mean(distribution_lengths_without_play), 'k', 'LineWidth', line_width)
plot(hmm_length_edges_centers,distribution_lengths_play, ':r')
plot(hmm_length_edges_centers,mean(distribution_lengths_play), 'r', 'LineWidth', line_width)

subplot(1,2,2)
plot(hmm_length_edges_centers,cum_distribution_lengths_without_play, ':k')
hold on
plot(hmm_length_edges_centers,mean(cum_distribution_lengths_without_play), 'k', 'LineWidth', line_width)
plot(hmm_length_edges_centers,cum_distribution_lengths_play, ':r')
plot(hmm_length_edges_centers,mean(cum_distribution_lengths_play), 'r', 'LineWidth', line_width)
print(gcf,'-vector','-dsvg',[figure_dir,'\hmm and play cum distributions.svg'])


%% load varaible onset to hmm


 variable_onset_struct_files = dir([last_hmm_dir,'\* variable_onset_struct*']);


variable_names = [];
total_number_of_hmm         = 0;
total_number_of_hmm_3states = 0;

for fn= 1:numel(variable_onset_struct_files)
     load([last_hmm_dir,'\',variable_onset_struct_files(fn).name]) 
        
     if fn==1
        
         variable_names = variable_onset_struct.variable_types';
     end
     total_number_of_hmm = total_number_of_hmm+size(variable_onset_struct.beh_properties_onset,2);
     total_number_of_hmm_3states = total_number_of_hmm_3states+size(variable_onset_struct.beh_properties_onset_3states  ,3);
end

all_variable_onsets= nan(numel(variable_names),total_number_of_hmm,size(variable_onset_struct.beh_properties_onset,3));
all_variable_offsets= nan(numel(variable_names),total_number_of_hmm,size(variable_onset_struct.beh_properties_onset,3));


all_hmm_lengths = [];
total_number_of_hmm         = 0;

for fn= 1:numel(variable_onset_struct_files)
      load([last_hmm_dir,'\',variable_onset_struct_files(fn).name]) 

        n_hmm = size(variable_onset_struct.beh_properties_onset,2);
      
        all_hmm_lengths = [all_hmm_lengths; diff(variable_onset_struct.filled_play_bouts')'];     
        variables_this_session = variable_onset_struct.beh_properties_onset ;
        for vn=1:size(variables_this_session,1)
            this_var = variables_this_session(vn,:,:);
            this_var = this_var(:);
            variables_this_session(vn,:,:) = (variables_this_session(vn,:,:) - mean(this_var, 'omitmissing'))/std(this_var, 'omitmissing');
        end
        all_variable_onsets(:,total_number_of_hmm+1:total_number_of_hmm+n_hmm,:)  =variables_this_session  ;
         
        variables_this_session = variable_onset_struct.beh_properties_offset ;
        for vn=1:size(variables_this_session,1)
            this_var = variables_this_session(vn,:,:);
            this_var = this_var(:);
           variables_this_session(vn,:,:) = (variables_this_session(vn,:,:)- mean(this_var, 'omitmissing'))/std(this_var, 'omitmissing');
        end
        all_variable_offsets(:,total_number_of_hmm+1:total_number_of_hmm+n_hmm,:)  = variables_this_session;
      
        total_number_of_hmm = total_number_of_hmm+n_hmm
end
psth_edges = variable_onset_struct.psth_edges;
%% select state with and witouh play and estiamte their lengths (needed to match length of engnaged and unegnaged)

is_there_play  =  any(is_this_hmm & is_there_play_beh,2);
there_is_no_play =~any(is_this_hmm & is_there_play_beh,2);

play_lengths    = all_hmm_lengths(is_there_play);
noplay_lengths  = all_hmm_lengths(there_is_no_play);

Cost = (play_lengths-noplay_lengths').^2;

%% find paried play and non pay states wih same length distributiomn (load if exist, estiamte otherwise)

if exist([last_hmm_dir,'\matching_lengths.mat'], 'file')~=2
    disp('Estimating')

 
costUnmatched =4;


figure
p = 0;

while p<0.05



    M = matchpairs(Cost,costUnmatched); %first colum is withplay second column witohuhtplay


    plot(play_lengths( M(:,1)), noplay_lengths( M(:,2)), '.')
    pause(.1)

    [h,p]= kstest2(play_lengths( M(:,1)), noplay_lengths( M(:,2)));
    costUnmatched = costUnmatched/1.05;
end

save([last_hmm_dir,'\matching_lengths.mat'], 'play_lengths','noplay_lengths','M','is_there_play','there_is_no_play')
else
     disp('Loading')
    load([last_hmm_dir,'\matching_lengths.mat'], 'play_lengths','noplay_lengths','M','is_there_play','there_is_no_play')
end
%% defined indexes of states with matched lengths
is_there_play = find(is_there_play);
is_there_play = is_there_play(M(:,1));


there_is_no_play = find(there_is_no_play);
there_is_no_play = there_is_no_play(M(:,2));




original_play = any(is_this_hmm & is_there_play_beh,2);
aux_false = false(size(original_play));
aux_false(is_there_play) = 1;
is_there_play = aux_false==1;

aux_false = false(size(original_play));
aux_false(there_is_no_play) = 1;
there_is_no_play = aux_false==1;

original_play(is_there_play)
original_play(there_is_no_play)

%% plot (select if needed) and time wrap varaibles aligned to hmm onset
 
plot_bool = false; %change this to true if you want to plot not timewraped variables
x_lim = [-2 4];
is_there_play  = is_there_play & all_hmm_lengths>0;
there_is_no_play = there_is_no_play & all_hmm_lengths>0;

[hmm_length_ordered, pb_order] = sort(all_hmm_lengths);

bin_size = mean(diff(psth_edges));
time_before_after = [-10 10];

wrapped_bins_porp = 1/3; %proportion of total amount of bins
Total       =  round(range(time_before_after)/((1-wrapped_bins_porp)*bin_size));
wrapped_n   = Total -round((range(time_before_after)/bin_size)) ;



time_wraped_varaibles_play = nan(size(all_variable_onsets,1), sum(is_there_play), Total);
time_wraped_varaibles_noplay = nan(size(all_variable_onsets,1), sum(there_is_no_play), Total);
time_wrapped_time = [-(time_before_after(1):bin_size:-bin_size)/time_before_after(1), linspace(0,1,wrapped_n ), 1+(bin_size:bin_size:time_before_after(2))/time_before_after(2)];

for variable_n=1:numel(variable_names)

   
    play_lengths = hmm_length_ordered(is_there_play(pb_order));
    no_play_lengths = hmm_length_ordered(there_is_no_play(pb_order));
    matrix2plot_withplay = squeeze(all_variable_onsets(variable_n,pb_order(is_there_play(pb_order)),:));
    matrix2plot_withoutplay = squeeze(all_variable_onsets(variable_n,pb_order(there_is_no_play(pb_order)),:));
     for j=1:size(matrix2plot_withplay,1)
         data_before = psth_edges>=time_before_after(1) & psth_edges<0;
         data_during = psth_edges>=0 & psth_edges<=play_lengths(j);
         data_after = psth_edges>play_lengths(j) & psth_edges<=play_lengths(j)+time_before_after(2);


         data_before = matrix2plot_withplay(j,data_before);
         time_during = psth_edges(data_during);
         data_during = matrix2plot_withplay(j,data_during);
         data_during = interp1(time_during, data_during,linspace(time_during(1), time_during(end), wrapped_n));
         data_after = matrix2plot_withplay(j,data_after);

         if numel(data_after)<round(time_before_after(2)/bin_size)
            data_after = [data_after, nan(1, round(time_before_after(2)/bin_size)-numel(data_after))];
         end

         time_wraped_varaibles_play(variable_n,j,:) = [data_before,data_during,data_after];
     end


      for j=1:size(matrix2plot_withoutplay,1)
         data_before = psth_edges>=time_before_after(1) & psth_edges<0;
         data_during = psth_edges>=0 & psth_edges<=play_lengths(j);
         data_after = psth_edges>play_lengths(j) & psth_edges<=play_lengths(j)+time_before_after(2);


         data_before = matrix2plot_withoutplay(j,data_before);
         time_during = psth_edges(data_during);
         data_during = matrix2plot_withoutplay(j,data_during);
         data_during = interp1(time_during, data_during,linspace(time_during(1), time_during(end), wrapped_n));
         data_after = matrix2plot_withoutplay(j,data_after);

         if numel(data_after)<round(time_before_after(2)/bin_size)
            data_after = [data_after, nan(1, round(time_before_after(2)/bin_size)-numel(data_after))];
         end

         time_wraped_varaibles_noplay(variable_n,j,:) = [data_before,data_during,data_after];
     end



  
    if plot_bool
        figure('units','normalized','outerposition',[0 0 .5 1]);
        colormap(1-gray)
        subplot(5,2,1)


        imagesc(psth_edges,1:numel(play_lengths), matrix2plot_withplay)
        hold on
        plot([0 0],[1 numel(play_lengths)],'r')
        hold on
        plot(play_lengths,1:numel(play_lengths),'r')
        axis xy
        yticks([])
        ylabel('With Play')
        xlim(x_lim)
        title(variable_names{variable_n})


        subplot(5,2,3)

        imagesc(psth_edges,1:numel(no_play_lengths), matrix2plot_withoutplay)
        hold on
        plot([0 0],[1 numel(no_play_lengths)],'r')
        hold on
        plot(no_play_lengths,1:numel(no_play_lengths),'r')
        axis xy
        xlim(x_lim)
        yticks([])
        ylabel('Without Play')


        subplot(5,2,5)
        mean2plot = mean(matrix2plot_withoutplay);
        [~, ~, ci] = ttest(matrix2plot_withoutplay);
        fill([psth_edges fliplr(psth_edges )], [ci(1,:) fliplr(ci(2,:)) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
        hold on
        plot(psth_edges,mean2plot, 'k' )
        mean2plot = mean(matrix2plot_withplay);
        [~, ~, ci] = ttest(matrix2plot_withplay);
        fill([psth_edges fliplr(psth_edges )], [ci(1,:) fliplr(ci(2,:)) ], 'r', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
        hold on
        plot(psth_edges,mean2plot, 'r' )
        xlim(x_lim)
        ylim tight
        y_lim = ylim;
        hold on
        plot([0 0],y_lim,'b', 'HandleVisibility','off')




        subplot(5,2,[7 9])
        mean2plot      = mean(matrix2plot_withoutplay);
        [~, ~, ci] = ttest(matrix2plot_withoutplay);
        for j=find(psth_edges>=0)
            lengths_to_include = no_play_lengths>=psth_edges(j);
            mean2plot(j) = mean(matrix2plot_withoutplay(lengths_to_include,j));
            ci(:,j) = mean2plot(j) + 1.96*std(matrix2plot_withoutplay(lengths_to_include,j))*[-1 1]/sqrt(sum(lengths_to_include));
        end
        no_nan = ~any(isnan(ci));
        fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,no_nan) fliplr(ci(2,no_nan)) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
        hold on
        plot(psth_edges,mean2plot, 'k' )

        mean2plot      = mean(matrix2plot_withplay);
        [~, ~, ci] = ttest(matrix2plot_withplay);
        for j=find(psth_edges>=0)
            lengths_to_include = play_lengths>=psth_edges(j);
            mean2plot(j) = mean(matrix2plot_withplay(lengths_to_include,j));
            ci(:,j) = mean2plot(j) + 1.96*std(matrix2plot_withplay(lengths_to_include,j))*[-1 1]/sqrt(sum(lengths_to_include));
        end
        no_nan = ~any(isnan(ci));
        fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,no_nan) fliplr(ci(2,no_nan)) ], 'r', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
        hold on
        plot(psth_edges,mean2plot, 'r' )
        y_lim2 = y_lim;
        y_lim2(2) = 1.5*y_lim(2);
        hold on
        plot([0 0],y_lim2,'b', 'HandleVisibility','off')
        axis tight
        legend({ 'Without Play','Play'})
        xlim(x_lim)



        subplot(5,2,2)
        matrix2plot_withplay = squeeze(all_variable_offsets(variable_n,pb_order(is_there_play(pb_order)),:));
        imagesc(psth_edges,1:numel(play_lengths), matrix2plot_withplay)
        hold on
        plot([0 0],[1 numel(play_lengths)],'r')
        hold on
        plot(-play_lengths,1:numel(play_lengths),'r')
        axis xy
        xlim(x_lim)
        yticks([])
        ylabel('With Play')
        title(variable_names{variable_n})

        subplot(5,2,4)
        matrix2plot_withoutplay = squeeze(all_variable_offsets(variable_n,pb_order(there_is_no_play(pb_order)),:));
        imagesc(psth_edges,1:numel(no_play_lengths), matrix2plot_withoutplay)
        hold on
        plot([0 0],[1 numel(no_play_lengths)],'r')
        hold on
        plot(-no_play_lengths,1:numel(no_play_lengths),'r')
        axis xy
        xlim(x_lim)
        yticks([])
        ylabel('Without Play')


        subplot(5,2,6)
        mean2plot = mean(matrix2plot_withoutplay);
        [~, ~, ci] = ttest(matrix2plot_withoutplay);
        fill([psth_edges fliplr(psth_edges )], [ci(1,:) fliplr(ci(2,:)) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
        hold on
        plot(psth_edges,mean2plot, 'k' )
        mean2plot = mean(matrix2plot_withplay);
        [~, ~, ci] = ttest(matrix2plot_withplay);
        fill([psth_edges fliplr(psth_edges )], [ci(1,:) fliplr(ci(2,:)) ], 'r', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
        hold on
        plot(psth_edges,mean2plot, 'r' )

        y_lim = ylim;
        hold on
        plot([0 0],y_lim,'b', 'HandleVisibility','off')
        axis tight
        xlim(x_lim)

        subplot(5,2,[8 10])
        mean2plot      = mean(matrix2plot_withoutplay);
        [~, ~, ci] = ttest(matrix2plot_withoutplay);
        for j=find(psth_edges<=0)
            lengths_to_include = no_play_lengths>=-psth_edges(j);
            mean2plot(j) = mean(matrix2plot_withoutplay(lengths_to_include,j));
            ci(:,j) = mean2plot(j) + 1.96*std(matrix2plot_withoutplay(lengths_to_include,j))*[-1 1]/sqrt(sum(lengths_to_include));
        end
        no_nan = ~any(isnan(ci));
        fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,no_nan) fliplr(ci(2,no_nan)) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
        hold on
        plot(psth_edges,mean2plot, 'k' )

        mean2plot      = mean(matrix2plot_withplay);
        [~, ~, ci] = ttest(matrix2plot_withplay);
        for j=find(psth_edges<=0)
            lengths_to_include = play_lengths>=-psth_edges(j);
            mean2plot(j) = mean(matrix2plot_withplay(lengths_to_include,j));
            ci(:,j) = mean2plot(j) + 1.96*std(matrix2plot_withplay(lengths_to_include,j))*[-1 1]/sqrt(sum(lengths_to_include));
        end
        no_nan = ~any(isnan(ci));
        fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,no_nan) fliplr(ci(2,no_nan)) ], 'r', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
        hold on
        plot(psth_edges,mean2plot, 'r' )
        y_lim2 = y_lim;
        y_lim2(2) = 1.5*y_lim(2);
        plot([0 0],y_lim2,'b', 'HandleVisibility','off')
        legend({ 'Without Play','Play'})
        axis tight
        xlim(x_lim)


        pause(.1)
    end
end


%% plot time wrapped data (PANEL H-J)
var_list = {'AnimalSpeed','NumCalls','RelativeDistance'};
variables2plot = find(ismember(variable_names,var_list))';
% time_wraped_varaibles_play 
% time_wraped_varaibles_noplay 
for vn=variables2plot
    figure('units','normalized','outerposition',[0 0 .2 1]);
    subplot(2,1,1)
    colormap(1-gray)
    imagesc(time_wrapped_time,1:(size(time_wraped_varaibles_play,2)*2), [squeeze(time_wraped_varaibles_noplay(vn,:,:));squeeze(time_wraped_varaibles_play(vn,:,:))] )
    axis xy
    hold on
    plot([time_wrapped_time(1) time_wrapped_time(end)], size(time_wraped_varaibles_play,2)*[1 1], 'r')
    title(variable_names{vn})
    subplot(2,1,2)
    hold on
    matrix2plot      = squeeze(time_wraped_varaibles_noplay(vn,:,:));
    [~, ~, ci] = ttest(matrix2plot);
    no_nan = ~any(isnan(ci));
    fill([time_wrapped_time(no_nan) fliplr(time_wrapped_time(no_nan) )], [ci(1,no_nan) fliplr(ci(2,no_nan)) ], k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    plot(time_wrapped_time,mean(matrix2plot), 'k' )

    matrix2plot      = squeeze(time_wraped_varaibles_play(vn,:,:));
    [~, ~, ci] = ttest(matrix2plot);
    no_nan = ~any(isnan(ci));
    fill([time_wrapped_time(no_nan) fliplr(time_wrapped_time(no_nan) )], [ci(1,no_nan) fliplr(ci(2,no_nan)) ], 'r', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    plot(time_wrapped_time,mean(matrix2plot), 'r' )


    print(gcf,'-vector','-dsvg',[figure_dir,'\', variable_names{vn}, ' between states tw.svg'])
    pause(.1)
end



%% prepare data for classification (dta for supp figure)
   hmm_type = any(is_this_hmm  & is_there_play_beh,2 );


hmm_data_mean = nan(numel(all_hmm_lengths), size(all_variable_onsets,1)+2);
hmm_data_all = [];
hmm_data_all_pre_post = [];
bout_index = [];
range2predict = -1;


   hmm_n = 1;
  
   for hmm_n=1:numel(all_hmm_lengths)
       disp(hmm_n)
        hmm_indexes     = is_this_hmm(hmm_n,:)==1;
        bout_index      = repmat(hmm_n,size(hmm_indexes));
        bout_index      = bout_index(hmm_indexes);
              
  
        hmm_data_all = [hmm_data_all;[squeeze(all_variable_onsets(:,hmm_n,hmm_indexes))' bout_index' repmat([all_hmm_lengths(hmm_n) hmm_type(hmm_n) ], sum(hmm_indexes), 1) is_there_play_beh(hmm_n,hmm_indexes)']];
         hmm_indexes     = is_this_hmm(hmm_n,:)==1;
         hmm_indexes(psth_edges<=0 & psth_edges>=range2predict) = 1;         
         bout_index = repmat(hmm_n,size(hmm_indexes));
          bout_index      = bout_index(hmm_indexes);
      hmm_data_all_pre_post = [hmm_data_all_pre_post;[squeeze(all_variable_onsets(:,hmm_n,hmm_indexes))' bout_index' repmat([all_hmm_lengths(hmm_n) hmm_type(hmm_n) ], sum(hmm_indexes), 1) is_there_play_beh(hmm_n,hmm_indexes)']];
   end

 hmm_data_all = array2table(hmm_data_all)  ;
 hmm_data_all.Properties.VariableNames = strsplit(num2str(1:size(hmm_data_all,2)),' ');
 hmm_data_all.Properties.VariableNames{end}     = 'PlayBool';
 hmm_data_all.Properties.VariableNames{end-1}   = 'StateType';
 hmm_data_all.Properties.VariableNames{end-2}   = 'StateLength';
 hmm_data_all.Properties.VariableNames{end-3}   = 'StateNum';


 hmm_data_all_pre_post = array2table(hmm_data_all_pre_post)  ;
 hmm_data_all_pre_post.Properties.VariableNames = strsplit(num2str(1:size(hmm_data_all_pre_post,2)),' ');
 hmm_data_all_pre_post.Properties.VariableNames{end}     = 'PlayBool';
 hmm_data_all_pre_post.Properties.VariableNames{end-1}   = 'StateType';
 hmm_data_all_pre_post.Properties.VariableNames{end-2}   = 'StateLength';
 hmm_data_all_pre_post.Properties.VariableNames{end-3}   = 'StateNum';


%% DAta used for trining SVM_play_90 (the remaining 90 is left for testing) , needed data for replicate figure is loaded in next section
%The original values used for training are saved in the matlab fil "SVM_play_90.mat"

sub_group_hmm = hmm_data_all(:,[1:14, 18]);
selected_values = randperm(size(sub_group_hmm,1), round(.9*size(sub_group_hmm,1)));

reimaining_events = ~ismember(1:size(sub_group_hmm,1),selected_values);
data4training = sub_group_hmm(selected_values,:);






%%  LOAD SVM_play_90 and associated data



  load([analyssis_folder, '\SVM_play_90.mat'],'SVM_play_90','selected_values','reimaining_events','sub_group_hmm','data4training')



%% use the traiend svm to score bins as more or less lickely to be play, this may take several minutes
[yfit,scores] = SVM_play_90.predictFcn(hmm_data_all);

play_song([],[], []);
% 
table2predict = hmm_data_all_pre_post(:, ismember(hmm_data_all_pre_post.Properties.VariableNames,list_of_Var));
[yfit_all,scores_all] = SVM_play_90.predictFcn(table2predict);
play_song([],[], []);
%% create socre figure (now assign each value to the corresponding raw in the hmm plot, SUpplementary Figure)

 score_values = nan(size(is_this_hmm));
  score_values_entire_matrix = nan(size(is_this_hmm));

re_buid_play =  zeros(size(is_this_hmm));

 for hmm_n=1:numel(all_hmm_lengths)
 
   
        hmm_indexes     = is_this_hmm(hmm_n,:)==1;
        re_buid_play(hmm_n,hmm_indexes) = hmm_data_all.PlayBool(hmm_data_all.StateNum==hmm_n);
            score_values(hmm_n,hmm_indexes) = scores(hmm_data_all.StateNum==hmm_n,2);

         hmm_indexes(psth_edges<=0 & psth_edges>=range2predict) = 1;         


   score_values_entire_matrix(hmm_n,hmm_indexes)  = scores_all(hmm_data_all_pre_post.StateNum==hmm_n,2);          
 end

%% plot play score for two different states (Supplementary Figure)
y_lim = [-2 1];
data2plot = score_values_entire_matrix;
slection_index = reimaining_events; %   here you select wich dataset to plot, "reimaining_events"
                                    % are the events that were not used for
                                    % training. if you keep just a true
                                    % array it will use all datapoints
                                    % (like on the commented line below)
 slection_index = true(size(reimaining_events));
                                  

% data2plot =score_values;
x_lim = [-1 3];
figure('units','normalized','outerposition',[0 0 .25 1]);
colormap(1-gray)
subplot(5,1,1)
bool_selection = is_there_play(pb_order) & slection_index(pb_order)';
matrix2plot_withplay = squeeze(data2plot(pb_order(bool_selection ),:));
imagesc(psth_edges,1:numel(hmm_length_ordered(bool_selection)), matrix2plot_withplay)
hold on
plot([0 0],[1 numel(hmm_length_ordered(bool_selection))],'r')
hold on
plot(hmm_length_ordered(bool_selection),1:numel(hmm_length_ordered(bool_selection)),'r')
axis xy
yticks([])
ylabel('With Play')
xlim(x_lim)

subplot(5,1,2)
bool_selection =there_is_no_play(pb_order) & slection_index(pb_order)';

matrix2plot_withoutplay = squeeze(data2plot(pb_order(bool_selection),:));
imagesc(psth_edges,1:numel(hmm_length_ordered(bool_selection)), matrix2plot_withoutplay)
hold on
plot([0 0],[1 numel(hmm_length_ordered(bool_selection))],'r')
hold on
plot(hmm_length_ordered(bool_selection),1:numel(hmm_length_ordered(bool_selection)),'r')
axis xy
xlim(x_lim)
yticks([])
ylabel('Without Play')


subplot(5,1,3:5)
mean2plot = mean(matrix2plot_withoutplay, 'omitmissing');
[~, ~, ci] = ttest(matrix2plot_withoutplay);
no_nan = ~any(isnan(ci));
fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,no_nan) fliplr(ci(2,no_nan)) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
hold on
plot(psth_edges,mean2plot, 'k' )
mean2plot = mean(matrix2plot_withplay, 'omitmissing');
[~, ~, ci] = ttest(matrix2plot_withplay);
no_nan = ~any(isnan(ci));
fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,no_nan) fliplr(ci(2,no_nan)) ], 'r', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')

hold on
plot(psth_edges,mean2plot, 'r' )
xlim(x_lim)
ylim(y_lim)
% y_lim = ylim;
hold on
plot([0 0],y_lim,'b', 'HandleVisibility','off')



%% classify hmm according to play score parameters (Supplementary Figure)

positve_scores = score_values;
% positve_scores(positve_scores<0) = NaN;

state_params = [mean(positve_scores,2, 'omitmissing') median(positve_scores,2, 'omitmissing') max(positve_scores,[],2, 'omitmissing') min(positve_scores,[],2, 'omitmissing')];


state_params_zscored  = [zscore(mean(score_values,2, 'omitmissing'))  zscore(median(score_values,2, 'omitmissing')) zscore(max(score_values,[],2, 'omitmissing') ) zscore(min(score_values,[],2, 'omitmissing'))];

[coeff,score_pca,latent,tsquared,explained,mu]  = pca(state_params_zscored); 



%% classify enganged and unenegaged states using mean, median, max and min play score duitng engaged and unengaged state (needed for Supplementay Figure)
state_params_zscored  = [zscore(mean(score_values,2, 'omitmissing'))  zscore(median(score_values,2, 'omitmissing')) zscore(max(score_values,[],2, 'omitmissing') ) zscore(min(score_values,[],2, 'omitmissing'))];

state_params_zscored = array2table(state_params_zscored);
state_params_zscored.Properties.VariableNames = {'mean' ,'median','max','min'  };
X  = state_params_zscored;
Y      =hmm_type;

k  = 10; % folds for corss validation
cv = cvpartition(Y, 'KFold', k);

withinConfMatrices = cell(k,1);
betweenConfMatrices = cell(k,1);
withinMetrics = zeros(k, 4);   % [TPR, FNR, FPR, TNR]
betweenMetrics = zeros(k, 4);  % same as above

for i = 1:k
    % -- Indices for fold i --
    trainIdx = training(cv, i);
    testIdx = test(cv, i);

    % === WITHIN-FOLD ===
    Mdl_within = fitcdiscr(X(trainIdx,:), Y(trainIdx));
    preds_within = predict(Mdl_within, X(testIdx,:));
    cm_within = confusionmat(Y(testIdx), preds_within);
    withinConfMatrices{i} = cm_within;

    % Extract binary classification metrics (assumes classes ordered properly)
    TP = cm_within(2,2); FN = cm_within(2,1);
    FP = cm_within(1,2); TN = cm_within(1,1);
    withinMetrics(i,:) = [TP/(TP+FN), FN/(TP+FN), FP/(FP+TN), TN/(FP+TN)];

    % === BETWEEN-FOLD ===
    j = mod(i, k) + 1;  % Next fold (cyclic) as test
    testIdx_b = test(cv, j);
    Mdl_between = fitcdiscr(X(trainIdx,:), Y(trainIdx));
    preds_between = predict(Mdl_between, X(testIdx_b,:));
    cm_between = confusionmat(Y(testIdx_b), preds_between);
    betweenConfMatrices{i} = cm_between;

    % Metrics
    TP = cm_between(2,2); FN = cm_between(2,1);
    FP = cm_between(1,2); TN = cm_between(1,1);
    betweenMetrics(i,:) = [TP/(TP+FN), FN/(TP+FN), FP/(FP+TN), TN/(FP+TN)];
end
cvPred = kfoldPredict(cvMdl);


%% now plot confussion matrices (Supp figure)



figure
ax = subplot(1,3,1);
plot(score_pca(hmm_type==1,1), score_pca(hmm_type==1,2), 'r.')
data = score_pca(hmm_type==1,[1 2]);
plot_2d_percentile_contours(data, 100*[.5 .78 .9], ax, 'r')
hold on
plot(score_pca(hmm_type==0,1), score_pca(hmm_type==0,2), 'k.')
data = score_pca(hmm_type==0,[1 2]);
plot_2d_percentile_contours(data, 100*[.5 .78 .9], ax, 'k')

subplot(1,3,2)


y_lim = [-2 2];
mean2plot = mean(matrix2plot_withoutplay, 'omitmissing');
[~, ~, ci] = ttest(matrix2plot_withoutplay);
no_nan = ~any(isnan(ci));
fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,no_nan) fliplr(ci(2,no_nan)) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
hold on
plot(psth_edges,mean2plot, 'k' )
mean2plot = mean(matrix2plot_withplay, 'omitmissing');
[~, ~, ci] = ttest(matrix2plot_withplay);
no_nan = ~any(isnan(ci));
fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,no_nan) fliplr(ci(2,no_nan)) ], 'r', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')

hold on
plot(psth_edges,mean2plot, 'r' )
xlim(x_lim)
ylim tight

hold on
plot([0 0],y_lim,'b', 'HandleVisibility','off')
ylim(y_lim)

subplot(1,3,3)
rand_x_pos = (rand(size(betweenMetrics))-.5)*.25;
plot(repmat(1:4,k,1)' + rand_x_pos',betweenMetrics(:, [1 2 4 3])', ':k')
hold on
plot(repmat(1:4,k,1)' + rand_x_pos',betweenMetrics(:, [1 2 4 3])', '.k', 'MarkerSize', 14)
xticks(1:4)
xlim([.5 4.5])
xticklabels({'TP','FN','TN','FP'})

%% save hmm type classification

print(gcf,'-vector','-dsvg',[figure_dir,'\ engaged vs unengaged classification.svg'])


%% Estiamte percentage of play within hmm (and shuffled percentages, not used anymore)


list_of_sessions = unique(session_index)';

percentage_of_play      = nan(numel(list_of_sessions),4);
schuffled_percentage    = nan(numel(list_of_sessions),4);

for sn = list_of_sessions



    percentage_of_play(sn,1) =   sum(sum(is_there_play_beh(session_index==sn,:) & is_there_hmm(session_index==sn,:)))/sum(sum(is_there_play_beh(session_index==sn,:)));
    percentage_of_play(sn,2) =   sum(sum(is_there_play_beh(session_index==sn,:) & ~is_there_hmm(session_index==sn,:)))/sum(sum(~is_there_play_beh(session_index==sn,:)));
    percentage_of_play(sn,3) =   sum(sum(~is_there_play_beh(session_index==sn,:) & ~is_there_hmm(session_index==sn,:)))/sum(sum(~is_there_play_beh(session_index==sn,:)));
    percentage_of_play(sn,4) =   sum(sum(~is_there_play_beh(session_index==sn,:) & is_there_hmm(session_index==sn,:)))/sum(sum(~is_there_play_beh(session_index==sn,:)));

    a = is_there_play_beh(session_index==sn,:) ;
    a_aux = a(randsample(numel(a), numel(a)));
    a = reshape(a_aux, size(a,1), size(a,2));
    b = is_there_hmm(session_index==sn,:);
     b_aux = b(randsample(numel(b), numel(b)));
    b = reshape(b_aux, size(b,1), size(b,2));


    schuffled_percentage(sn,1) =   sum(sum(a & b ))/sum(sum(a));
    schuffled_percentage(sn,2) =   sum(sum(a & ~b ))/sum(sum(a));
    schuffled_percentage(sn,3) =   sum(sum(~a & ~b ))/sum(sum(~a));
    schuffled_percentage(sn,4) =   sum(sum(~a & b ))/sum(sum(~a));

end



%% load confussion matrix (not used anymore)


confusion_matrix_files = dir([last_hmm_dir, '\*confusion_matrix*']);

all_confusion_matrix = nan(numel(confusion_matrix_files),4);

for fn= 1:numel(confusion_matrix_files)
    load([last_hmm_dir,'\',confusion_matrix_files(fn).name])
all_confusion_matrix(fn,:) = confusion_matrix;
end

figure



subplot(1,2,1)

x_toplot = repmat([1 2 3 4], size(percentage_of_play,1),1) + .25*(rand(size(percentage_of_play))-.5);
plot(x_toplot, 100*percentage_of_play, 'k.')
hold on
plot(x_toplot', 100*percentage_of_play', 'k:')
plot([1 2 3 4], 100*mean(percentage_of_play), '_r', 'MarkerSize',10)
title('% of play that is hmm')


subplot(1,2,2)

x_toplot = repmat([1 2 3 4], size(percentage_of_play,1),1) + .25*(rand(size(percentage_of_play))-.5);
plot(x_toplot, 100*all_confusion_matrix, 'k.')
hold on
plot(x_toplot', 100*all_confusion_matrix', 'k:')
plot([1 2 3 4], 100*mean(all_confusion_matrix), '_r', 'MarkerSize',10)
title('% of hmm that is play')
print(gcf,'-vector','-dsvg',[figure_dir,'\hmm and play overlap.svg'])



%% old panel G (commented)
% x_lim = [-5 10];
% 
% figure('units','normalized','outerposition',[0 0 .5 1]);
% % current_hmm =  tripple_states==3;
% current_hmm = is_this_hmm;
% filled_hmm_states = filled_play_bouts;
% [hmm_length_ordered, pb_order] = sort(diff(filled_hmm_states'));
% shorter = diff(filled_hmm_states')<1.5
% longer  = diff(filled_hmm_states')>1.5
% properly_labeled = filled_hmm_states(pb_order,1)<=Inf;
% 
% 
% 
% subplot(5,1,1:3)
% imagesc(psth_edges, 1:sum(properly_labeled), 1-cat(3,is_there_hmm(pb_order(properly_labeled),:), is_there_play_beh(pb_order(properly_labeled),:), is_there_play_beh(pb_order(properly_labeled),:) ))
% axis xy
% hold on
% plot([0 0], [.5 sum(properly_labeled)+.5], 'k')
% plot(hmm_length_ordered,1:sum(properly_labeled),'k')
% yticks([])
% xlim(x_lim)
% xticks(psth_edges(1):psth_edges(end))
% title('Indivudual Play behaviors')
% 
% subplot(5,1,4:5)
% plot(psth_edges, mean(is_there_play_beh), 'k')
% hold on
% plot(psth_edges, mean(is_there_play_beh(shorter,:))', 'b')
% plot(psth_edges, mean(is_there_play_beh(longer,:)), 'r')
% xlim(x_lim)

%% panel G (with play)

% Assume binary matrices A, B, C of the same size
[m, n] = size(A);

% Define RGB colors
color1 = [1 .2 .2];       % onlyAC (red)
color2 = [.75 .75 .75];     % onlyAnoC (dark red)
color3 = [.5 0 0];       % B==1 (pink)
color4 = [1 1 1];       % noneAB (white)

% Initialize RGB image
RGB = ones(m, n, 3);  % default white

% Logical masks
onlyAC    = (A==1) & ~(B==1) & (C==1);
onlyAnoC  = (A==1) & ~(B==1) & ~(C==1);
B_is_1    = (B==1);          % replaces previous bothAB
noneAB    = ~(A | B);

% Assign colors
for ch = 1:3
    RGB(:,:,ch) = ...
        color1(ch) * onlyAC + ...
        color2(ch) * onlyAnoC + ...
        color3(ch) * B_is_1 + ...
        color4(ch) * noneAB;
end

% Plot
figure('units','normalized','outerposition',[0 0 .5 1]);
colormap(gray)
[hmm_length_ordered, pb_order] = sort(diff(filled_hmm_states'));
imagesc(psth_edges,1:size(RGB,1), RGB(pb_order,:,:));
axis xy
xlim([-5 10])


figure('units','normalized','outerposition',[.5 0 .5 1]);
grayImage = rgb2gray(RGB);
colormap(gray)
[hmm_length_ordered, pb_order] = sort(diff(filled_hmm_states'));
imagesc(psth_edges,1:size(RGB,1), grayImage(pb_order,:,:));
axis xy
xlim([-5 10])



%% save  figure  hmm play overlap
print(gcf,'-vector','-dsvg',[figure_dir,'\all hmm onset w play.svg'])


%% alternative panel G without play

% Assume binary matrices A, B, C of the same size
[m, n] = size(A);

% Define RGB colors
color1 = [1 0 0];       % onlyAC (red)
color2 = [.75 .75 .75];     % onlyAnoC (dark red)
color3 = [.5 0 0];       % B==1 (pink)
color4 = [1 1 1];       % noneAB (white)

% Initialize RGB image
RGB = ones(m, n, 3);  % default white

% Logical masks
onlyAC    = (A==1)   & (C==1);
onlyAnoC  = (A==1)   & ~(C==1);
B_is_1    = (B==1)*0;          % replaces previous bothAB
noneAB    = ~(A );

% Assign colors
for ch = 1:3
    RGB(:,:,ch) = ...
        color1(ch) * onlyAC + ...
        color2(ch) * onlyAnoC + ...
        color3(ch) * B_is_1 + ...
        color4(ch) * noneAB;
end

% Plot
figure('units','normalized','outerposition',[0 0 .5 1]);
[hmm_length_ordered, pb_order] = sort(diff(filled_hmm_states'));
image(psth_edges,1:size(RGB,1), RGB(pb_order,:,:));
axis xy
xlim([-5 10])
%% now time_wrapped_version (preparing PANEL L)

% Make sure time_val is a row vector
time_val = psth_edges(:)'; % Convert to 1 x n_timepoints if needed
% Duration bounds
% Parameters
T1 = -5;  % Pre-event duration (e.g., seconds)
T2 =  5;  % Post-event duration
hmm_lengths = diff(filled_hmm_states');
val2use = hmm_lengths<max(psth_edges)-T2 & hmm_lengths>0;
hmm_lengths = hmm_lengths(val2use);
play_matrix = is_there_play_beh(val2use,:);

n_bins = 50;

% Number of trials/samples
n_trials = size(play_matrix, 1);
n_timepoints = size(play_matrix, 2);

% Make sure time_val is a row vector matching columns of is_there_play_beh
time_val = time_val(:)';

% Preallocate output matrix: rows = trials, cols = 60 interpolated time points
mz = zeros(n_trials, n_bins * 3);
last_play_latency = nan(numel(n_trials),1);
next_play_latency = nan(numel(n_trials),1);

for trial_i = 1:n_trials
    event_length = hmm_lengths(trial_i);
    play_starts   = psth_edges(find(diff(play_matrix(trial_i, :))==1));
    play_ends    =  psth_edges(find(diff(play_matrix(trial_i, :))==-1));
   

    last_Event      = max(play_ends(play_ends<0));
    following_Event = min(play_starts(play_starts>event_length))-event_length;
    if isempty(last_Event)        
        last_Event  = NaN;
    end
    if isempty(following_Event)        
        following_Event  = NaN;
    end

    last_play_latency(trial_i) = last_Event;
    next_play_latency(trial_i) = following_Event;

    % Define the three intervals for this trial
    pre_interval = [T1, 0];
    event_interval = [0, event_length];
    post_interval = [event_length, event_length + T2];

    % Extract indices and data for pre-event
    idx_pre = find(time_val >= pre_interval(1) & time_val <= pre_interval(2));
    t_pre_orig = time_val(idx_pre);
    signal_pre = play_matrix(trial_i, idx_pre);
    t_pre_interp = linspace(pre_interval(1), pre_interval(2), n_bins);
    interp_pre = interp1(t_pre_orig, signal_pre, t_pre_interp, 'linear', 'extrap');

    % Extract indices and data for event
    idx_event = find(time_val >= event_interval(1) & time_val <= event_interval(2));
    t_event_orig = time_val(idx_event);
    signal_event = play_matrix(trial_i, idx_event);
    t_event_interp = linspace(event_interval(1), event_interval(2), n_bins);
    interp_event = interp1(t_event_orig, signal_event, t_event_interp, 'linear', 'extrap');

    % Extract indices and data for post-event
    idx_post = find(time_val >= post_interval(1) & time_val <= post_interval(2));
    t_post_orig = time_val(idx_post);
    signal_post = play_matrix(trial_i, idx_post);
    t_post_interp = linspace(post_interval(1), post_interval(2), n_bins);
    interp_post = interp1(t_post_orig, signal_post, t_post_interp, 'linear', 'extrap');

    % Concatenate interpolated parts (each length 20)
    mz(trial_i, :) = [interp_pre, interp_event, interp_post];
end

[~, lenght_order] = sort(hmm_lengths);


%% PANEL L

mz_play = sum(mz(:, (n_bins+1):(2*n_bins)),2)>0;
figure
n = 256; % Number of colors in the colormap

% Create a colormap that starts at white [1 1 1] and ends at red [1 0 0]
redColormap = [linspace(1,.5,n)' linspace(1,0,n)' linspace(1,0,n)'];

% Apply the colormap
colormap(redColormap);

mz(mz>1) = 1;
hmm_with_play = ceil(mz(mz_play,:));
hmm_length_with_play = hmm_lengths(mz_play);
[~,lenght_order_with_play] = sort(hmm_length_with_play);
session_index_play          = session_index(mz_play);

hmm_without_play        = ceil(mz(~mz_play,:));
hmm_length_without_play = hmm_lengths(~mz_play);
[~,lenght_order_without_play] = sort(hmm_length_without_play);
session_index_without_play      = session_index(~mz_play); 

subplot(5,1,1:2)
imagesc(1:size(hmm_with_play,2), 1:size(mz,1), ceil([hmm_without_play(lenght_order_without_play,:);hmm_with_play(lenght_order_with_play,:)]))
clim([0 1])
hold on
xticks([0:3]*n_bins)
plot([0 n_bins*3], size(hmm_without_play,1)*[1 1], 'r', 'LineWidth',2)
yticks([size(hmm_without_play,1)/2 size(hmm_without_play,1)+size(hmm_with_play,1)/2])
ptcg_wo_play    = round(100*size(hmm_without_play,1)/size(mz,1));
ptcg_with_play  = round(100*size(hmm_with_play,1)/size(mz,1));
yticklabels({num2str(ptcg_wo_play), num2str(ptcg_with_play)})
xticklabels([])
plot([n_bins n_bins], [1 size(mz,1)], 'c')
plot([2*n_bins 2*n_bins], [1 size(mz,1)], 'c')
set(gca, 'TickDir', 'out')

axis xy

ax = subplot(5,1,3:5)

session_list = unique(session_index_play)';
staked_mean_play = nan(numel(session_list),size(mz,2));

for sn = session_list
    staked_mean_play(sn,:) = mean(hmm_with_play(session_index_play==sn,:));
end

plot(1:size(hmm_with_play,2),100*staked_mean_play, ':k')
hold on
plot(1:size(hmm_with_play,2),100*mean(staked_mean_play), 'k', 'LineWidth',2)
hold on


session_list = unique(session_index_without_play)';
staked_mean_without_play = nan(numel(session_list),size(mz,2));
for sn = session_list
    staked_mean_without_play(sn,:) = mean(hmm_without_play(session_index_without_play==sn,:));
end
% second_red = [153 51 51]/255;
second_red = 'c';
plot(1:size(hmm_without_play,2), 100*staked_mean_without_play, ':', 'Color', second_red)
hold on
plot(1:size(hmm_with_play,2),100*mean(staked_mean_without_play), 'Color',second_red, 'LineWidth',2)
set(ax, 'TickDir', 'out')

%% save figure hmm and play peri and within state
print(gcf,'-vector','-dsvg',[figure_dir,'\hmm and play peri and within state probablitycorrect colors.svg'])


%% load behavior aligned to hmm



behavior_onset_offset_struct_files = dir([last_hmm_dir, '\*behavior_onset_offset_struct*']);


behavior_type_list = [];
total_number_of_hmm = 0;
total_number_of_hmm_3states = 0;

for fn= 1:numel(behavior_onset_offset_struct_files)
    load([last_hmm_dir, '\',behavior_onset_offset_struct_files(fn).name])
    behavior_type_list = [behavior_type_list; behavior_onset_offset_struct.behavior_tpes];
    total_number_of_hmm = total_number_of_hmm+size(behavior_onset_offset_struct.filled_play_bouts,1);
    total_number_of_hmm_3states = total_number_of_hmm_3states++size(behavior_onset_offset_struct.filled_hmm_3states,1);

end
behavior_type_list = unique(behavior_type_list);
behavior_type_list(ismember(behavior_type_list, {'Partners session', 'SA'})) = [];


merged_behaviors_onset  = zeros(numel(behavior_type_list),total_number_of_hmm,size(behavior_onset_offset_struct.behavior_offset,3));


merged_behaviors_onset_3states  = zeros(3,numel(behavior_type_list),total_number_of_hmm_3states,size(behavior_onset_offset_struct.behavior_offset,3));
all_hmm_3states = [];
all_hmm_lengths = [];
total_number_of_hmm = 0;
total_number_of_hmm_3states = 0;
session_index_again = [];
for fn= 1:numel(behavior_onset_offset_struct_files)
      disp([ 'Loading ', behavior_onset_offset_struct_files(fn).name])
    load([last_hmm_dir, '\',behavior_onset_offset_struct_files(fn).name])
    disp('Processing')
  
    n_hmm = size(behavior_onset_offset_struct.filled_play_bouts,1);
    n_hmm_3states = size(behavior_onset_offset_struct.filled_hmm_3states,1);
    all_hmm_lengths = [all_hmm_lengths; diff(behavior_onset_offset_struct.filled_play_bouts')'];
    session_index_again = [session_index_again;ones(size(behavior_onset_offset_struct.filled_play_bouts,1),1)*fn];
    this_3_States_matrix = behavior_onset_offset_struct.filled_hmm_3states;

    for j=1:3
        this_3_States_matrix (behavior_onset_offset_struct.filled_hmm_3states(:,1)==j-1,1) = re_assignment(fn,j)-1;
    end
    all_hmm_3states = [all_hmm_3states;this_3_States_matrix];

    current_behaviors = find(ismember(behavior_type_list,behavior_onset_offset_struct.behavior_tpes));

    for beavior_present = current_behaviors'
        beh_index = ismember(behavior_onset_offset_struct.behavior_tpes,behavior_type_list(beavior_present));
        merged_behaviors_onset(beavior_present,total_number_of_hmm+1:total_number_of_hmm+n_hmm,:) = behavior_onset_offset_struct.behavior_onset(beh_index,:,:);

        merged_behaviors_onset_3states(:,beavior_present,total_number_of_hmm_3states+1:total_number_of_hmm_3states+n_hmm_3states,:) = behavior_onset_offset_struct.behavior_onset_3states(re_assignment(fn,:), beh_index,:,:);


    end
    total_number_of_hmm=total_number_of_hmm+n_hmm;
    total_number_of_hmm_3states = total_number_of_hmm_3states+n_hmm_3states;

end

%% now make timewrap

% Make sure time_val is a row vector
time_val = psth_edges(:)'; % Convert to 1 x n_timepoints if needed
% Duration bounds
% Parameters
T1 = -5;  % Pre-event duration (e.g., seconds)
T2 =  5;  % Post-event duration
hmm_lengths = diff(filled_hmm_states');
val2use = hmm_lengths<max(psth_edges)-T2 & hmm_lengths>0;
hmm_lengths = hmm_lengths(val2use);
play_behavior_matrix = merged_behaviors_onset(:,val2use,:);

n_bins = 50;

% Number of trials/samples
n_behaviors     = size(play_behavior_matrix, 1) ;
n_trials        = size(play_behavior_matrix, 2);
n_timepoints    = size(play_behavior_matrix, 3);

% Make sure time_val is a row vector matching columns of is_there_play_beh
time_val = time_val(:)';

% Preallocate output matrix: rows = trials, cols = 60 interpolated time points
mz_behavior = zeros(n_behaviors,n_trials, n_bins * 3);
for bn = 1:n_behaviors
    disp(bn)
    for trial_i = 1:n_trials
        event_length = hmm_lengths(trial_i);




        % Define the three intervals for this trial
        pre_interval = [T1, 0];
        event_interval = [0, event_length];
        post_interval = [event_length, event_length + T2];

        % Extract indices and data for pre-event
        idx_pre = find(time_val >= pre_interval(1) & time_val <= pre_interval(2));
        t_pre_orig = time_val(idx_pre);
        signal_pre = squeeze(play_behavior_matrix(bn,trial_i, idx_pre));
        t_pre_interp = linspace(pre_interval(1), pre_interval(2), n_bins);
        interp_pre = interp1(t_pre_orig, signal_pre, t_pre_interp, 'linear', 'extrap');

        % Extract indices and data for event
        idx_event = find(time_val >= event_interval(1) & time_val <= event_interval(2));
        t_event_orig = time_val(idx_event);
        signal_event = squeeze(play_behavior_matrix(bn,trial_i, idx_event));
        t_event_interp = linspace(event_interval(1), event_interval(2), n_bins);
        interp_event = interp1(t_event_orig, signal_event, t_event_interp, 'linear', 'extrap');

        % Extract indices and data for post-event
        idx_post = find(time_val >= post_interval(1) & time_val <= post_interval(2));
        t_post_orig = time_val(idx_post);
        signal_post = squeeze(play_behavior_matrix(bn,trial_i, idx_post));
        t_post_interp = linspace(post_interval(1), post_interval(2), n_bins);
        interp_post = interp1(t_post_orig, signal_post, t_post_interp, 'linear', 'extrap');

        % Concatenate interpolated parts (each length 20)
        mz_behavior(bn,trial_i, :) = [interp_pre, interp_event, interp_post];
    end
end
%% now plot ( never used)
behaviors2merge = {'Pounce_A','Pounce_B';'Pounce_Ai','Pounce_Bi' };

for bn = 1:size(behaviors2merge,1)
    beh_index =  find(ismember(behavior_type_list,behaviors2merge(bn,:)));
    for sub_bn = 1:numel(beh_index)
    mz_behavior(beh_index(sub_bn),:,:) = sum( mz_behavior(beh_index,:,:)) ;
    end
end

play_bool = sum(is_this_hmm & is_there_play_beh,2)>0;
behaviors2plot = {'Pounce_A','Pounce_Ai','CB','CC','CD','Pin','Boxing','Bite','Sniffing','Rearing','Escape','Grooming', 'Evasion', 'Scratch'};


beh_index = ismember(behavior_type_list,behaviors2plot);
mean_prob = squeeze(mean(mz_behavior(beh_index,play_bool,:),2, "omitmissing"));
sub_behavior_labes = behavior_type_list(beh_index);
for j=1:size(mean_prob,1)
    mean_prob(j,:) = movmean(mean_prob(j,:),10);
end




[norm_prob, max_loc] = max(mean_prob,[],2);
[latency_val, latency_order] = sort(max_loc);
sub_behavior_labes = sub_behavior_labes(latency_order);
norm_mean_prob = diag(1./norm_prob)*mean_prob;
figure
subplot(2,3,1)
colormap(1-gray)
index = latency_val<n_bins;
imagesc(1:size(norm_mean_prob,2), 1:sum(index), norm_mean_prob(latency_order(index),:))
axis xy

subplot(2,3,4)
plot(1:size(norm_mean_prob,2),  norm_mean_prob(latency_order(index),:) + repmat((1:sum(index))',1,size(norm_mean_prob,2)), 'k')
yticks((1:sum(index)) +.5)
yticklabels(sub_behavior_labes(index))
hold on
plot(repmat([1 3*n_bins],sum(index)+1,1)', repmat((1:sum(index)+1)',1,2)', ':k')
plot([n_bins n_bins],[1 sum(index)+1], 'k')
plot(2*[n_bins n_bins],[1 sum(index)+1], 'k')

subplot(2,3,2)
colormap(1-gray)
index = latency_val>=n_bins & latency_val<=2*n_bins;
imagesc(1:size(norm_mean_prob,2), 1:sum(index), norm_mean_prob(latency_order(index),:))
axis xy

subplot(2,3,5)
plot(1:size(norm_mean_prob,2),  norm_mean_prob(latency_order(index),:) + repmat((1:sum(index))',1,size(norm_mean_prob,2)), 'r')
yticks((1:sum(index)) +.5)
yticklabels(sub_behavior_labes(index))
hold on
plot(repmat([1 3*n_bins],sum(index)+1,1)', repmat((1:sum(index)+1)',1,2)', ':k')
plot([n_bins n_bins],[1 sum(index)+1], 'k')
plot(2*[n_bins n_bins],[1 sum(index)+1], 'k')

subplot(2,3,3)
colormap(1-gray)
index = latency_val>2*n_bins;
imagesc(1:size(norm_mean_prob,2), 1:sum(index), norm_mean_prob(latency_order(index),:))
axis xy

subplot(2,3,6)
plot(1:size(norm_mean_prob,2),  norm_mean_prob(latency_order(index),:) + repmat((1:sum(index))',1,size(norm_mean_prob,2)), 'm')
yticks((1:sum(index)) +.5)
yticklabels(sub_behavior_labes(index))
hold on
plot(repmat([1 3*n_bins],sum(index)+1,1)', repmat((1:sum(index)+1)',1,2)', ':k')
plot([n_bins n_bins],[1 sum(index)+1], 'k')
plot(2*[n_bins n_bins],[1 sum(index)+1], 'k')


%%  Possible Figure G (behavior onset to engaged  states)

figure
ax = subplot(1,1,1)
plot(1:size(norm_mean_prob,2),  norm_mean_prob(latency_order,:) + repmat((1:numel(latency_order))',1,size(norm_mean_prob,2)), 'r')
hold on
plot(repmat([1 3*n_bins],numel(latency_order)+1,1)', repmat((1:numel(latency_order)+1)',1,2)', ':k')
plot([n_bins n_bins],[1 numel(latency_order)+1], 'k')
plot(2*[n_bins n_bins],[1 numel(latency_order)+1], 'k')
yticks((1:numel(latency_order)) +.5)
yticklabels(sub_behavior_labes)
axis tight
set(ax, 'TickDir', 'out')
%% save figure
print(gcf,'-vector','-dsvg',[figure_dir,'\hmm and behavior onset offset.svg'])


%% analysie time serieos of hmm
bin_size = 0.01;
time2check = 90;

is_this_a_playful_hmm = sum(is_this_hmm & is_there_play_beh,2)>0;


sn = 1;
this_session_hmm = filled_play_bouts(session_index==sn,:);
this_session_playful_hmm = is_this_a_playful_hmm(session_index==sn);

all_time = (floor(this_session_hmm(1,1)/bin_size)*bin_size):bin_size:(ceil(this_session_hmm(end,2)/bin_size)*bin_size);

playful_time_series         = any(all_time>=this_session_hmm(this_session_playful_hmm,1) & all_time<=this_session_hmm(this_session_playful_hmm,2));
non_playful_time_series     = any(all_time>=this_session_hmm(~this_session_playful_hmm,1) & all_time<=this_session_hmm(~this_session_playful_hmm,2));


n_lags = round(time2check/bin_size);
[play_acf, lags] = autocorr(double(playful_time_series), 'NumLags',n_lags);
% [non_play_acf, ~] = autocorr(double(non_playful_time_series), 'NumLags',n_lags);


figure
plot(lags*bin_size,play_acf )



%% create shufled distribution
figure('units','normalized','outerposition',[0 0 1 1]);
n_rand = 1000;
bin_size = 0.01;
time2check = 120;
n_lags                      = round(time2check/bin_size);
all_play_ac                 = nan(numel(unique(session_index)), n_lags+1);
all_upper_bound95           = nan(numel(unique(session_index)), n_lags+1);
all_upper_bound99           = nan(numel(unique(session_index)), n_lags+1);
pcrtl_play                  = nan(numel(unique(session_index)), n_lags+1);
random_peaks_distribution   = [];

for sn = unique(session_index)'

    this_session_hmm            = filled_play_bouts(session_index==sn,:);
    this_session_playful_hmm    = is_this_a_playful_hmm(session_index==sn);
    all_time                    = (floor(this_session_hmm(1,1)/bin_size)*bin_size):bin_size:(ceil(this_session_hmm(end,2)/bin_size)*bin_size);

    playful_time_series         = any(all_time>=this_session_hmm(this_session_playful_hmm,1) & all_time<=this_session_hmm(this_session_playful_hmm,2));


  
    [play_acf, lags] = autocorr(double(playful_time_series), 'NumLags',n_lags);

    all_play_ac(sn,:) = play_acf;


    shufflled_play_acf = nan(n_rand,n_lags+1);
    for nr = 1:n_rand
        re_order = randsample(size(this_session_hmm,1),size(this_session_hmm,1));



        this_session_hmm = filled_play_bouts(session_index==sn,:);
        this_session_intervals = this_session_hmm(2:end,1)-this_session_hmm(1:end-1,2);
        this_session_intervals = [this_session_intervals;mean(this_session_intervals)];
        this_session_playful_hmm = is_this_a_playful_hmm(session_index==sn);

        shuffled_hmm = this_session_hmm(re_order,:);
        shuffled_hmm_lengths = diff(shuffled_hmm')';
        shuffled_intervals = this_session_intervals(re_order);
        shuffled_playful = this_session_playful_hmm(re_order,:);

        shuffled_hmm(1,1) = this_session_hmm(1,1);
        for j=1:numel(shuffled_hmm_lengths)-1

            shuffled_hmm(j,2)   = shuffled_hmm(j,1) + shuffled_hmm_lengths(j);
            shuffled_hmm(j+1,1) = shuffled_hmm(j,1) + shuffled_hmm_lengths(j) + shuffled_intervals(j);
        end
        shuffled_hmm(end,2 ) =  shuffled_hmm(end,1) + shuffled_hmm_lengths(end);
       all_time                    = (floor(shuffled_hmm(1,1)/bin_size)*bin_size):bin_size:(ceil(shuffled_hmm(end,2)/bin_size)*bin_size);

        shuffled_playful_time_series         = any(all_time>=shuffled_hmm(shuffled_playful,1) & all_time<=shuffled_hmm(shuffled_playful,2));

        [shufflled_play_acf(nr,:), ~] = autocorr(double(shuffled_playful_time_series), 'NumLags',round(time2check/bin_size));
    end

    



    matrix2plot = shufflled_play_acf;
    this_pctl = 100 * mean(repmat(play_acf,n_rand,1)<= shufflled_play_acf);


     for nr = 1:25:n_rand

         shufled_pctl =  100 * mean(repmat(shufflled_play_acf(nr,:),n_rand,1)<= shufflled_play_acf);
         y = 100  -shufled_pctl;
        y = smoothdata(y,'gaussian',10/bin_size);
        [pks,locs,w,p] = findpeaks(y,'MinPeakHeight',80,'MinPeakProminence', .4*range(y), 'MinPeakDistance',10/bin_size);
        random_peaks_distribution =  [random_peaks_distribution; [(lags(locs)*bin_size)'  (lags(locs)*bin_size)'*0+sn (lags(locs)*bin_size)'*0+nr]];
     end
    



    pcrtl_play(sn,:) = this_pctl;
    upper_bound95 = prctile(matrix2plot, 95, 1);   % 5th percentile across rows (dim 1)
    upper_bound99 = prctile(matrix2plot, 99, 1);


    subplot(5,3, sn)
    plot(lags*bin_size,play_acf , 'r')
    hold on
    plot(lags*bin_size, upper_bound95, 'b')
    plot(lags*bin_size, upper_bound99, 'g')


    all_upper_bound95(sn,:) =upper_bound95;
    all_upper_bound99(sn,:) = upper_bound99;
    xlim([0 20])
    pause(.1)
   

end
%% save autocorrelgoram examples (PLAY BOUT FIGURE A)
print(gcf,'-vector','-dsvg',[figure_dir,'\all autocorrelgoram examples.svg'])



%% plot autocorrelogram excess play (PLAY BOUT FIGURE B)
excces_play_time    = 5;
x_lim               = [0 10];
y_lim_pctl          = [75 100];
y_lim_p_val         = [.9 1];

figure
colormap(jet)
subplot(5 ,2,[1 3 5])
imagesc(lags*bin_size, 1:15,all_play_ac-all_upper_bound95)
axis xy
clim([-.2 .2])
xlim(x_lim)

rank_signficiant = nan(1, size(all_play_ac,2));
for t=1:numel(rank_signficiant)
rank_signficiant(t) = signrank(all_play_ac(:,t),all_upper_bound95(:,t), 'tail','right');
end

subplot(5,2,[7 9])
plot(x_lim, [0 0], 'k')
xlim(x_lim)
hold on
plot(lags*bin_size,all_play_ac-all_upper_bound95, ':k')
plot(lags*bin_size,mean(all_play_ac-all_upper_bound95), 'k', 'LineWidth',3)
xlim(x_lim)
ylim([-.2 .2])

yyaxis right
semilogy(lags*bin_size,1-rank_signficiant, 'Color',  [0.5, 0, 0.5])
hold on
semilogy(lags*bin_size,lags*0 + 1-0.05, ':', 'Color',  [0.5, 0, 0.5])
ylim(y_lim_p_val)

subplot(5 ,2,[1 3 5]+1)
imagesc(lags*bin_size, 1:15,100-pcrtl_play)
clim(y_lim_pctl)
axis xy
xlim(x_lim)


subplot(5,2,[7 9]+1)
plot(x_lim, [95 95], 'k')
hold on
plot(lags*bin_size,100-pcrtl_play, ':k')
plot(lags*bin_size,100-mean(pcrtl_play), 'k', 'LineWidth',3)
matrix2plot = 100-pcrtl_play;
ci = [-1; 1]*std(matrix2plot)/sqrt(size(matrix2plot,1)) + [1;1]*mean(matrix2plot);
fill([lags*bin_size fliplr(lags*bin_size)], [ci(1,:) fliplr(ci(2,:))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
xlim(x_lim)
ylim(y_lim_pctl)
%% save figure
 print(gcf,'-vector','-dsvg',[figure_dir,'\autocorrelgoram stats.svg'])

%% findint second peaks 
figure
all_time_events = [];
for sn=1:15
y = 100  -pcrtl_play(sn,:);
y = smoothdata(y,'gaussian',10/bin_size);
[pks,locs,w,p] = findpeaks(y,'MinPeakHeight',80,'MinPeakProminence', .4*range(y), 'MinPeakDistance',10/bin_size);





subplot(5,3,sn)
plot(lags*bin_size,y)
hold on
plot(lags(locs)*bin_size, y(locs), '.r')

all_time_events =[all_time_events; [(lags(locs)*bin_size)'  (lags(locs)*bin_size)'*0+sn]];
end

%% plot distributiuomn of second play peak and random distribution (PLAY BOUT FIGURE C)


[real_f, real_x] = histcounts(all_time_events(:,1), 0:1:120,'Normalization','cdf');  

n_sessions = unique(random_peaks_distribution(:,2));
distribution_random = nan(numel(n_sessions), numel(real_f))
Xboots = cell(numel(n_sessions),1);
for nn =1:numel(n_sessions)
    Xboots{nn} =random_peaks_distribution(random_peaks_distribution(:,2)==nn,1); 
[random_f, random_x] = histcounts(random_peaks_distribution(random_peaks_distribution(:,2)==nn,1), 0:1:120,'Normalization','cdf'); 
distribution_random(nn,:) =random_f;
end


figure
bin_centers = .5*(real_x(1:end-1) + real_x(2:end));
plot(bin_centers, real_f, 'r')
% f = CDF values, x = sorted data
hold on
plot(bin_centers, distribution_random, 'k')
xlabel('Value'); ylabel('CDF');
% title('Empirical CDF');
% histogram(all_time_events(:,1), 10:10:150, 'Normalization','cdf')
% hold on
% histogram(random_peaks_distribution(:,1), 10:10:150, 'Normalization','cdf')

%% estiamte significance (take a bit of time) and add o pvalue (PLAY BOUT FIGURE C)


X_all = cell2mat(Xboots(:));  % Concatenate all bootstrap samples
 y =all_time_events(:,1);
% --- Compute observed KS statistic ---
[~,~,ks_stat] = kstest2(X_all, y);

% --- Permutation test ---
nPerm = 5000; % Number of permutations
combined = [X_all(:); y(:)];
nX = numel(X_all);
nY = numel(y);
perm_stats = zeros(nPerm,1);

for i = 1:nPerm
    idx = randperm(nX + nY);
    x_perm = combined(idx(1:nX));
    y_perm = combined(idx(nX+1:end));
    [~,~,perm_stats(i)] = kstest2(x_perm, y_perm);
end

% --- Compute p-value ---
p_value = mean(perm_stats >= ks_stat);

title(p_value)
%% save figure
print(gcf,'-vector','-dsvg',[figure_dir,'\second play after 1 minute .svg'])
%% Load embedings and play beahviors


hmm_data_folder                 = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\HMM data\HMM raw data';
labeled_data_folder             = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\HMM data\locomotive behaviors';
segmented_data_folder           =  '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\locomotive behaviors 2 partners';
not_labeled_data_folder         =  '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\non labeled behavior 2 partners';
folder_with_anima_names         = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\OLD DATA FORMAT';
behavior_classification_folder  = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\Behavior classification';
dir_list = dir(folder_with_anima_names);
dir_list(1:2) = [];
spatial_property_names      = {  'Speed','AngleSpeed','AngleAcc','Acc','Wall2CenterPos'...
    'RelativeDistance','RelativeSpeed','RelativeAngleSpeed','RelativeAngleAcc', 'RelativeAcc'};
call_prop_list = {'PrincipalFrequencykHz', 'SlopekHzs', 'Sinuosity', 'DeltaFreqkHz', 'FrequencyStandardDeviationkHz'};


behaviors2check = {'Pin', 'Boxing', 'Evasion', 'Pounce_A','Pounce_B','CD','Escape','CC','CB','Pounce_Ai','Pounce_Bi','Rearing','Sniffing', 'Bite', 'Scratch', 'Grooming'}; %% here you decide what behavior to extract
%% load data behavior data

folder2save = segmented_data_folder;

load([folder2save,'/all_behavior.mat'],'all_behavior')

load([folder2save,'/all_not_labeled_behavior.mat'],'all_not_labeled_behavior')
load([folder2save,'/all_var_names.mat'],'all_var_names')

%% load embedings (only for CNN trianed data, not used at the end)
% embedding_folder = not_labeled_data_folder;
% all_embedings_CNN = readNPY([embedding_folder,'\','embeddings.npy']);
% % filename = 'embedding_file_order.txt';
% filename = 'file_order.txt';
% fileList = readlines([embedding_folder,'\',filename]);
% 
% % Initialize array to hold numeric IDs
% numFiles = length(fileList);
% fileNumbers = zeros(numFiles, 1);
% 
% % Extract numeric part from each filename
% for i = 1:numFiles
%     name = fileList(i);
%     % Remove '.npy' and convert to number
%     fileNumbers(i) = str2double(erase(name, '.npy'));
% end
% 
% 
% [~,correct_order] = sort(fileNumbers);
% 
% all_embedings_CNN = all_embedings_CNN(correct_order,:);
%% estimating mean properties per behavior (used for trainign classificator)


all_properties = [ all_behavior(:,1);all_not_labeled_behavior(:,1)];
all_mean_prop = [];
for j=1:size(all_properties,1)
    this_var = all_properties{j,1};
    if size(this_var,1)==1
        all_mean_prop = [all_mean_prop;[this_var size(this_var,1) sum(this_var(:,1))]];
    else
        all_mean_prop = [all_mean_prop;[mean(this_var,'omitmissing') size(this_var,1) sum(this_var(:,1))]];
    end
    behavior_length(j) =size(this_var,1);
end
all_mean_prop(:, end-1) = zscore(all_mean_prop(:, end-1));
all_mean_prop(:, end) = (all_mean_prop(:, end) - mean(all_mean_prop(:, end), 'omitmissing'))/std(all_mean_prop(:, end), 'omitmissing');
all_spatial_prop = all_var_names(1:numel(all_var_names)-numel(call_prop_list)-1);
call_prop    = all_mean_prop(:, numel(all_spatial_prop)+1: numel(all_spatial_prop)+numel(call_prop_list)+1);
spatial_prop = all_mean_prop(:, 1:numel(all_spatial_prop));



[coeff_call,score_call,latent_call,tsquared_call,explained_call,mu_call]  = pca(call_prop);
[coeff_spatial,score_spatial,latent_spatial,tsquared_spatial,explained_spatial,mu_spatial]  = pca(spatial_prop);
%% creat behavioral labels array

behavior_labels = [all_behavior(:,2);repmat({''},size(all_not_labeled_behavior,1),1)];

%% estimating  umap (using the mean of varaibles as features)
% data2umap= double(all_embedings_CNN);
data2umap = all_mean_prop;
data2umap(:, ismember(all_var_names, 'WallPos')) = [];
nan_values_cnn = any(isnan(data2umap),2);

data2umap = data2umap(~nan_values_cnn,:);
matched_labels = behavior_labels(~nan_values_cnn,:);



n_dim = 2;
% [reduction, umap] = run_umap(data2classify, 'n_components',2 );
[embedding_umaps, umapStruct] = run_umap(data2umap, ...
    'n_components',  n_dim, ...
    'n_neighbors', 25, ...
    'min_dist', 0.1, ...
    'metric', 'euclidean');




%% estimating umpa using mean properties
% variables_90 = [score_call(:, 1:min(find(ceil(cumsum(explained_call))>90))) score_spatial(:, 1:min(find(ceil(cumsum(explained_spatial))>90)))];
% data2_properties_umap = all_mean_prop;
% data2_properties_umap(:,5) = [];
% nan_pca_variables = any(isnan(data2_properties_umap), 2);
% [pca_umaps, pca_umaps_STruct] = run_umap(data2_properties_umap(~nan_pca_variables,:), ...
%     'n_components',  n_dim, ...
%     'n_neighbors', 25, ...
%     'min_dist', 0.01, ...
%     'metric', 'euclidean');


% behavior_length = zscore(behavior_length);
%% estimate  play,bite, reading model model


play_var = {'CC','Escape', 'CB'};
aggresive_var = {'Bite'};
rearing_var = {'Grooming','Scratch','Rearing'};

play_labels = (all_behavior(:,2));
play_labels(ismember(play_labels, 'Pounce_B')) = {'Pounce_A'};
play_labels(ismember(play_labels, 'Pounce_Bi')) = {'Pounce_Ai'};
all_behavior_labels = play_labels;


play_labels(ismember(play_labels,play_var)) = {'Play'};
play_labels(ismember(play_labels,aggresive_var)) = {'Aggression'};
play_labels(ismember(play_labels,rearing_var)) = {'Regulation'};


only_play_prop = all_mean_prop(ismember(play_labels, {'Play','Aggression', 'Regulation'}),:);
only_play_labels    = play_labels(ismember(play_labels, {'Play','Aggression', 'Regulation'}));

only_play_prop = array2table(only_play_prop);

n = size(only_play_prop,2);
colNames = arrayfun(@(i) sprintf('prop_%d', i), 1:n, 'UniformOutput', false);
only_play_prop.Properties.VariableNames = colNames;
only_play_prop(:,5) = [];



%%
load([behavior_classification_folder,'\LDA_PLAY_AGG_REG.mat'])
trained_model = LDA_PLAY_AGG_REG.ClassificationDiscriminant;
Features = all_mean_prop;   
Features(:,5) = []; %removing wall pos
n = size(Features,2);
Features = array2table(Features);
colNames = arrayfun(@(i) sprintf('mean_prop_%d', i), 1:n, 'UniformOutput', false);
Features.Properties.VariableNames = colNames;

Features = Features(~nan_pca_variables,:);
umap_pca = pca_umaps;
% Assume you already have:
% trained_model: your trained LDA model
% Features: your feature matrix (N x d)
% umap_pca: your UMAP/PCA embedding (N x 2)
% trained_model.ClassNames: cell array of class names

% 1. Get LDA predictions and posterior probabilities
[predictedLabels, posteriorProbs] = predict(trained_model, Features);

% 2. Get max posterior probability for each observation (proximity to closest centroid)
[maxProb, ~] = max(posteriorProbs, [], 2);

% 3. Map categorical predicted class to numeric for colormap
[~, classIdx] = ismember(predictedLabels, trained_model.ClassNames);

% 4. Create a custom colormap for the 3 classes
colors = lines(numel(trained_model.ClassNames)); % distinct colors for each class\
% colors = hsv(numel(trained_model.ClassNames));
pointColors = zeros(size(classIdx,1),3);
for i = 1:numel(trained_model.ClassNames)
    % Blend class color with white based on confidence (maxProb)
    pointColors(classIdx==i,:) = (1-maxProb(classIdx==i))*[1 1 1] + maxProb(classIdx==i)*colors(i,:);
end

% 5. Plot
figure;
scatter(umap_pca(:,1), umap_pca(:,2), 40, pointColors, 'filled');
xlabel('UMAP/PCA 1');
ylabel('UMAP/PCA 2');
title('LDA Class Predictions in UMAP/PCA Space');
grid on;

% Add legend for classes
hold on;
for i = 1:numel(trained_model.ClassNames)
    scatter(nan, nan, 80, colors(i,:), 'filled', 'DisplayName', trained_model.ClassNames{i});
end
legend('show');

%% plot clusters of behavior sun embedign space (using umap or pca FIGURE 1 D)

behaviors2check = {'Pin' ,'Boxing' ,'Evasion' ,'Pounce_A' ,'Pounce_B','CD'  ,'Escape','CC'  ,'CB'  ,'Pounce_Ai' ,'Pounce_Bi' ,'Rearing' ,'Sniffing' ,'Bite' , 'Scratch' ,'Grooming',''}; %% here you decide what behavior to extract


varaibles2cluster = num2cell(1:16);
varaibles2cluster = {[10 11]}
cluster_cloors = repmat({[0 0 1]},1,numel(varaibles2cluster));
% dimension2project = [score_call(:,1), score_spatial(:,1)];
dimension2project= embedding_umaps;

pcts = [0.1:0.1:0.8];
  axiis_lim = [-13    7.75   -3.5    7.3]
for nc = 1:numel(varaibles2cluster)
  figure
  
    scatter(dimension2project(:,1), dimension2project(:,2), 20, '.k');
    hold on
    index = ismember(matched_labels,behaviors2check(varaibles2cluster{nc})) ;
    X = dimension2project(index,:);
    % Define grid over which to evaluate the density
    x1 = linspace(min(dimension2project(:,1))-1, max(dimension2project(:,1))+1, 100);
    x2 = linspace(min(X(:,2))-1, max(X(:,2))+1, 100);
    [xg, yg] = meshgrid(x1, x2);
    grid_points = [xg(:), yg(:)];

    % Perform 2D Kernel Density Estimation
    [f, xi] = ksdensity(X, grid_points);  % f is the density at each grid point
    f_grid = reshape(f, length(x2), length(x1)); % reshape to grid

    % Normalize to get cumulative density
    f_sorted = sort(f(:), 'descend');
    cdf = cumsum(f_sorted) / sum(f_sorted);

    % Define percentile levels (in density units)
    % desired percentiles
    levels = zeros(size(pcts));
    for i = 1:length(pcts)
        idx = find(cdf >= pcts(i), 1, 'first');
        levels(i) = f_sorted(idx);
    end
    % contour(x1, x2, f_grid, sort(levels), 'LineWidth', .1, 'EdgeColor',cluster_cloors{nc});

    sorted_levels = sort(levels);
    for j=fliplr(1:numel(levels)-1)
        contourf(x1, x2, f_grid, sorted_levels([j j+1]), 'LineWidth', .5, 'FaceColor',cluster_cloors{nc}, 'FaceAlpha',.4*j/numel(levels),  'EdgeColor','None');
    end
    % axis(axiis_lim)

    title(behaviors2check(varaibles2cluster{nc}))
    print(gcf,'-vector','-dsvg',[figure_dir,'\', behaviors2check{nc}, ' cnn plot.svg'])
end




%% plot pca umap with color code
% 6. Plot
figure;

[~,re_order] = sort(posteriorProbs(:,3), 'descend');
scatter(umap_pca(re_order,1), umap_pca(re_order,2), 40, posteriorProbs(re_order,:)*colors, 'filled', 'MarkerEdgeColor','none','MarkerFaceAlpha',0.25);
title('Smoothed Posterior Map (Masked to Data Region)');
xlabel('UMAP/PCA 1'); ylabel('UMAP/PCA 2');

%% save figure

print(gcf,'-vector','-dsvg',[figure_dir,'\umap pca colorplot stats.svg'])
%% sort play by score value


trained_model = LDA_PLAY_AGG_REG.ClassificationDiscriminant;
Features = all_mean_prop(1:size(all_behavior,1),:);   
Features(:,5) = [];
n = size(Features,2);
Features = array2table(Features);
colNames = arrayfun(@(i) sprintf('mean_prop_%d', i), 1:n, 'UniformOutput', false);
Features.Properties.VariableNames = colNames;

% Features = Features(~nan_pca_variables,:);
umap_pca = pca_umaps;
% Assume you already have:
% trained_model: your trained LDA model
% Features: your feature matrix (N x d)
% umap_pca: your UMAP/PCA embedding (N x 2)
% trained_model.ClassNames: cell array of class names

% 1. Get LDA predictions and posterior probabilities
[predictedLabels, posteriorProbs] = predict(trained_model, Features);

classMeans = trained_model.Mu;          % C x D
Sigma = trained_model.Sigma;            % D x D (pooled covariance)
classes = trained_model.ClassNames;
nClasses = size(classMeans,1);

% Compute between-class scatter
overallMean = mean(classMeans,1);
B = zeros(size(Sigma));
for k = 1:nClasses
    diff = classMeans(k,:) - overallMean;
    B = B + (diff' * diff);
end

% Solve for discriminant directions (eigenvectors of inv(Sigma)*B)
[W,~] = eig(pinv(Sigma)*B);
W = real(W);   % remove tiny imaginary parts
% Sort by eigenvalues (descending)
[~, idx] = sort(diag(eig(pinv(Sigma)*B)), 'descend');
W = W(:, idx);

% Project features into this space
Xproj = (Features{:,:} - overallMean) * W;

% Get class centroids in projected space
projMeans = (classMeans - overallMean) * W;


i = 1; j = 2; % classes
v = projMeans(j,:) - projMeans(i,:);
scores = ((Xproj - projMeans(i,:)) * v') ./ (norm(v)^2);
scores = max(0,min(1,scores)); % clamp to [0,1]

%%
all_behavior_relabeled = all_behavior(:,2);
all_behavior_relabeled(ismember(all_behavior_relabeled, 'Pounce_B')) = {'Pounce_A'};
all_behavior_relabeled(ismember(all_behavior_relabeled, 'Pounce_Bi')) = {'Pounce_Ai'};
behavior_list = unique(all_behavior(:,2))';
mean_val_per_category = nan(numel(behavior_list), 1);
behavior_class = nan(size(scores,1),1);

for bn = 1:numel(behavior_list)
    mean_val_per_category(bn) = mean(scores(ismember(all_behavior_relabeled, behavior_list{bn})));
    behavior_class(ismember(all_behavior_relabeled, behavior_list{bn})) = bn;
end
[sorted_means, scores_sorted] = sort(mean_val_per_category, 'desc');

new_order = zeros(size(scores_sorted));
new_order(scores_sorted) = 1:numel(behavior_list);
behavior_class_sorted = new_order(behavior_class);

% 4. Re-map behavior_class to this new order


behavior_labels = all_behavior_relabeled;
x_pos = behavior_class_sorted + (rand(size(behavior_class_sorted))-.5)*.5;
%% plotng beavhir according to theri score in the play-bite sapce
figure
subplot(1,2,1)
swarmchart(behavior_class_sorted,scores, '.k')
hold on
plot(1:numel(sorted_means), sorted_means, '_r', 'MarkerSize', 10)
xticks(1:numel(behavior_list))
xticklabels(behavior_list(scores_sorted))


figure
[~,re_order] = sort(scores);
scatter(x_pos(re_order), scores(re_order), 15, scores(re_order)*[1 0 0] + (1-scores(re_order))*[0 0 1], 'filled', 'MarkerFaceAlpha',1);
hold on
scatter(1:numel(sorted_means), sorted_means, 'ko', 'filled')
plot(1:numel(sorted_means), sorted_means, 'k', 'LineWidth',2)
xticks(1:numel(behavior_list))
xticklabels(behavior_list(scores_sorted))
axis tight

figure
[~,re_order] = sort(posteriorProbs(:,3), 'descend');
scatter(x_pos(re_order), scores(re_order), 15, posteriorProbs(re_order,:)*colors, 'filled', 'MarkerFaceAlpha',.5);
hold on
% plot(sort([(1:(numel(sorted_means)))-.5,(1:numel(sorted_means))+.5 ]), sorted_means(sort([1:numel(sorted_means),1:numel(sorted_means) ])), 'k');
% plot([(1:(numel(sorted_means)))-.5;(1:numel(sorted_means))-.5 ], [0*sorted_means'; sorted_means'], 'k')
% plot([(1:(numel(sorted_means)))+.5;(1:numel(sorted_means))+.5 ], [0*sorted_means'; sorted_means'], 'k')
scatter(1:numel(sorted_means), sorted_means, 'ko', 'filled')
plot(1:numel(sorted_means), sorted_means, 'k', 'LineWidth',2)
xticks(1:numel(behavior_list))
xticklabels(behavior_list(scores_sorted))
xlim tight
%%
print(gcf,'-vector','-dsvg',[figure_dir,'\Play_Bite scale.svg'])

%% show how this score realted to locomotion and calls

figure
            x2plot =score_spatial(1:numel(scores),1);
            y2plot = score_call(1:numel(scores),1);
            [~,descending_order] = sort(scores, 'descend');

scatter(x2plot(descending_order)-min(x2plot)+0.001, y2plot(descending_order)-min(y2plot)+0.001, 20, posteriorProbs(descending_order,:)*colors, 'filled', 'MarkerFaceAlpha',.25);
set(gca,'xscale','log')
set(gca,'yscale','log')
xlabel('Locomotive PC')
ylabel('Call PC')
axis tight

%% save score call and locomotion relation

print(gcf,'-vector','-dsvg',[figure_dir,'\score to locomotion and call.svg'])
% scatter(x2plot, y2plot, 20, posteriorProbs*colors, 'filled', 'MarkerFaceAlpha',.25);
%%

figure

distribution_range = [0:0.001:1]


swarmchart(categorical(all_behavior(:,2)),(posteriorProbs(:,2)-posteriorProbs(:,1))./(posteriorProbs(:,2)+posteriorProbs(:,1)))


%% estimating  confussion matrices
load([behavior_classification_folder,'\trainedModel_LDA_ALL_BEHAVIORS.mat'],'trainedModel_LDA_ALL_BEHAVIORS') 

trained_model = trainedModel_LDA_ALL_BEHAVIORS.ClassificationDiscriminant;

cvMdl = trained_model.crossval;

k = cvMdl.KFold;
class_names =  trained_model .ClassNames;
cm_folds = nan(k,numel(class_names),numel(class_names));
sim    = cm_folds;
 n = numel(class_names);
for i = 1:k
    testIdx = test(cvMdl.Partition, i);
    preds = predict(cvMdl.Trained{i}, cvMdl.X(testIdx,:));
    CM = confusionmat(cvMdl.Y(testIdx), preds);    
    cm_folds(i,:,:) = CM;  


end
mean_cm = squeeze(mean(cm_folds));


row_sums = sum(mean_cm, 1);
CM_perc = mean_cm ./ row_sums;   % fraction per row
CM_perc = CM_perc * 100;
%%
figure
colormap(1-gray)
CM_perc(1:n+1:end) = 0;
imagesc(class_names,class_names,CM_perc)
hold on
for j=1:n

    this_class_conf = CM_perc(:,j);
    [top_pctg, top_2] = sort(this_class_conf, 'descend');
    top_2 = top_2(1:2);
    top_pctg = top_pctg(1:2);
for k=1:2
    text(j-.3,top_2(k),num2str(round(top_pctg(k),1)),'Color','r')
end
end


colorbar

%%  save confusion matrix (sUpp figu)
print(gcf,'-vector','-dsvg',[figure_dir,'\ confussion matrix.svg'])

%%
% trained LDA classificator to classify each behavior using mean properties
% of the behaviors, EXCLUDING tigmotaxis.
trained_model = trainedModel_LDA_ALL_BEHAVIORS.ClassificationDiscriminant;
centroids = trained_model.Mu;  % size: [numClasses x numFeatures]
covMat = trained_model.Sigma;  % shared covariance matrix

% Compute pairwise Mahalanobis distances between centroids
numClasses = size(centroids,1);
D = zeros(numClasses);
for i = 1:numClasses
    for j = 1:numClasses
        diff = centroids(i,:) - centroids(j,:);
        D(i,j) = sqrt(diff / covMat * diff');  % Mahalanobis distance
    end
end

% Use MDS to project these distances into 2D
Ymds = mdscale(D, 2, 'Criterion', 'metricstress');

% Plot the behaviors in this 2D space
figure;
scatter(Ymds(:,1), Ymds(:,2), 80, 'filled');
text(Ymds(:,1)+0.02, Ymds(:,2), trained_model.ClassNames, 'FontSize', 12);
title('Behaviors in LDA-centroid space (MDS of Mahalanobis distances)');
xlabel('Dimension 1');
ylabel('Dimension 2');
axis equal; grid on;

%% save distane between centroids (Supp Figure)
print(gcf,'-vector','-dsvg',[figure_dir,'\centroid distance.svg'])

%%

prop_names = [all_var_names 'duration', 'length'];
prop_names(ismember(prop_names,'WallPos')) = [];

Xfull =all_mean_prop;
Xfull(:,5) = [];

Xtraining = trainedModel_LDA_ALL_BEHAVIORS.ClassificationDiscriminant.X; % if it's cross-validated


[~, trainRows_90] = ismember(Xfull,Xtraining{:,:}, 'rows');
cvPred = kfoldPredict(cvMdl);
cm = confusionmat(Y, cvPred);

%%


%%
olny_play_variables_90 = variables_90(ismember(play_labels, {'Play','Aggression', 'Regulation'}),:);

olny_play_variables_90 = array2table(olny_play_variables_90);
n = size(olny_play_variables_90,2);
colNames = arrayfun(@(i) sprintf('mean_prop_%d', i), 1:n, 'UniformOutput', false);
olny_play_variables_90.Properties.VariableNames = colNames;


%%
X = variables_90;
X_table = array2table(X, 'VariableNames', colNames);
model2use = LDA_mean_prop_CHASING_BITE;
% model2use = LDA_BITE_REARING_CHASING
 [yfit,scores] = model2use.predictFcn(X_table);


%% load model if saved
load([folder2save,'\trainedModel_CHASING_BITE.mat'], 'trainedModel_CHASING_BITE')

 embedings_all_behavior_with_length = [all_embedings_CNN,behavior_length];
n = size(embedings_all_behavior_with_length,2);
colNames = arrayfun(@(i) sprintf('embedding_%d', i), 1:n, 'UniformOutput', false);
X = embedings_all_behavior_with_length;

X_table = array2table(X, 'VariableNames', colNames);
X_table(:,7) = [];
 [yfit,scores] = trainedModel_CHASING_BITE.predictFcn(X_table);
% embedding_umaps_labeled = embedding_umaps;
% figure
% hold on
% for j=1:size(embedding_umaps_labeled,1)
%     if ~any(isnan(scores(j,:)))
%         plot(embedding_umaps_labeled(j,1), embedding_umaps_labeled(j,2), '.', 'Color', [0 0 1]*scores(j,1) +[1 0 0]*scores(j,2))
%     end
% end

%% scatter plot play score using cnn umap
figure
hold on
no_nan_scores = scores(~nan_values_cnn,:);

for j=1:size(embedding_umaps,1)
    if ~isnan(no_nan_scores(j,2))
        plot(embedding_umaps(j,1), embedding_umaps(j,2), '.', 'Color', [0 0 1]*no_nan_scores(j,1) +[1 0 0]*no_nan_scores(j,2))
    end
end


%% scatter plot ussing  mean properites
figure
hold on
no_nan_scores = scores(~nan_pca_variables,:);

for j=1:size(pca_umaps,1)
    if ~isnan(no_nan_scores(j,2))
        plot(pca_umaps(j,1), pca_umaps(j,2), '.', 'Color', [0 0 1]*no_nan_scores(j,1) +[1 0 0]*no_nan_scores(j,2))
    end
end




%% heat matp  umap


dimension2project  =pca_umaps;
grid_size = 500;  % resolution of heatmap
radius = .5;    % neighborhood radius for averaging
x = dimension2project(:,1);
y = dimension2project(:,2);
cat_val = ismember(yfit, 'Play');
% Create grid
x_edges = linspace(min(x), max(x), grid_size);
y_edges = linspace(min(y), max(y), grid_size);
[Xg, Yg] = meshgrid(x_edges, y_edges);
heatmap_vals = nan(size(Xg));

% Compute average categorical value in a neighborhood for each grid cell
for i = 1:numel(Xg)
    dx = x - Xg(i);
    dy = y - Yg(i);
    dist = sqrt(dx.^2 + dy.^2);
    neighbors = dist < radius;
    if any(neighbors)
        heatmap_vals(i) = mean(cat_val(neighbors));
    else
        heatmap_vals(i) = NaN;  % leave empty if no points
    end
end


heatmap_vals = imgaussfilt(heatmap_vals, 4);
% --- Plot heatmap ---
figure;
colormap(jet)
pcolor(x_edges, y_edges, heatmap_vals);
hold on
% plot(dimension2project(:,1), dimension2project(:,2), 'k.')
shading flat
axis xy;  % correct orientation
colorbar;


%% ploting paly behaviors in umap space using filled contours

dimension2project  =embedding_umaps;
cluster_cloors = {'b', 'r'}
varaibles2cluster = unique(yfit);
pcts = [0.1:0.05:.6];
  figure
for nc = 1:numel(varaibles2cluster)
  
    
    scatter(dimension2project(:,1), dimension2project(:,2), 20, '.k');
    hold on
    index = ismember(yfit,varaibles2cluster{nc});
    index = index(~nan_values_cnn);
    X = dimension2project(index,:);
    % Define grid over which to evaluate the density
    x1 = linspace(min(dimension2project(:,1))-1, max(dimension2project(:,1))+1, 100);
    x2 = linspace(min(X(:,2))-1, max(X(:,2))+1, 100);
    [xg, yg] = meshgrid(x1, x2);
    grid_points = [xg(:), yg(:)];

    % Perform 2D Kernel Density Estimation
    [f, xi] = ksdensity(X, grid_points);  % f is the density at each grid point
    f_grid = reshape(f, length(x2), length(x1)); % reshape to grid

    % Normalize to get cumulative density
    f_sorted = sort(f(:), 'descend');
    cdf = cumsum(f_sorted) / sum(f_sorted);

    % Define percentile levels (in density units)
    % desired percentiles
    levels = zeros(size(pcts));
    for i = 1:length(pcts)
        idx = find(cdf >= pcts(i), 1, 'first');
        levels(i) = f_sorted(idx);
    end
    sorted_levels = sort(levels);
    for j=fliplr(1:numel(levels)-1)
    contourf(x1, x2, f_grid, sorted_levels([j j+1]), 'LineWidth', .5, 'FaceColor',cluster_cloors{nc}, 'FaceAlpha',.4*j/numel(levels),  'EdgeColor','None');
    end
    title(varaibles2cluster{nc})
        axis(axiis_lim)

end
%% save figure
print(gcf,'-vector','-dsvg',[figure_dir,'\cnn plot.svg'])

%% ploting play behavior in reduced space using non filled contours


behaviors2check = {'Pin' ,'Boxing' ,'Evasion' ,'Pounce_A' ,'Pounce_B','CD'  ,'Escape','CC'  ,'CB'  ,'Pounce_Ai' ,'Pounce_Bi' ,'Rearing' ,'Sniffing' ,'Bite' , 'Scratch' ,'Grooming',''}; %% here you decide what behavior to extract


varaibles2cluster = num2cell(1:numel(behaviors2check));
varaibles2cluster = {[4 5]}
cluster_cloors = repmat({[1 0 0]},1,numel(varaibles2cluster));
% dimension2project = [score_call(:,1), score_spatial(:,1)];
% dimension2project= pca_umaps;
dimension2project = pca_umaps;

pcts = [0.1:0.1:0.5];
    figure
for nc = 1:numel(varaibles2cluster)
  subplot(4,4,nc)
    scatter(dimension2project(:,1), dimension2project(:,2), 20, '.k');
    hold on
    index = ismember(matched_labels,behaviors2check(varaibles2cluster{nc})) ;
    X = dimension2project(index,:);
    % Define grid over which to evaluate the density
    x1 = linspace(min(dimension2project(:,1))-1, max(dimension2project(:,1))+1, 100);
    x2 = linspace(min(X(:,2))-1, max(X(:,2))+1, 100);
    [xg, yg] = meshgrid(x1, x2);
    grid_points = [xg(:), yg(:)];

    % Perform 2D Kernel Density Estimation
    [f, xi] = ksdensity(X, grid_points);  % f is the density at each grid point
    f_grid = reshape(f, length(x2), length(x1)); % reshape to grid

    % Normalize to get cumulative density
    f_sorted = sort(f(:), 'descend');
    cdf = cumsum(f_sorted) / sum(f_sorted);

    % Define percentile levels (in density units)
    % desired percentiles
    levels = zeros(size(pcts));
    for i = 1:length(pcts)
        idx = find(cdf >= pcts(i), 1, 'first');
        levels(i) = f_sorted(idx);
    end
    contour(x1, x2, f_grid, sort(levels), 'LineWidth', .1, 'EdgeColor',cluster_cloors{nc});
    title(behaviors2check{nc})
end
%% estiamte LDA for  all behaviors


embedings_behavior = all_embedings_CNN(1:size(all_behavior,1),:);
embedings_behavior_with_length = [all_embedings_CNN(1:size(all_behavior,1),:),behavior_length(1:size(all_behavior,1))];

behavior_human_labels = (all_behavior(:,2));
behavior_human_labels(ismember(behavior_human_labels, 'Pounce_B')) = {'Pounce_A'};
behavior_human_labels(ismember(behavior_human_labels, 'Pounce_Bi')) = {'Pounce_Ai'};
behavior_human_labels = categorical(behavior_human_labels);





X = embedings_behavior_with_length;

X_table = array2table(X, 'VariableNames', colNames);
X_table(:,7) = [];
 [yfit,scores] = trainedModel_CHASING_BITE.predictFcn(X_table);

  [numVals, categoryNames] = grp2idx(behavior_human_labels);



   %%

   print(gcf,'-vector','-dsvg',[figure_dir,'\play axis cnn.svg'])


%% now estimating play no play  hmm aaxis (bar plot as on old hmm estiamtes)

 last_hmm_dir = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\HMM 2 and 3 states 2 partners';


play_behavior_struct_files = dir([last_hmm_dir, '\*play_behavior_struct*']);




all_behavior_types = [];

for fn= 1:numel(play_behavior_struct_files)
     load([last_hmm_dir, '\',play_behavior_struct_files(fn).name]) 
     all_behavior_types = [all_behavior_types; play_behavior_struct.behavior_types];

end
all_behavior_types = unique(all_behavior_types);

all_numbers         = zeros(numel(all_behavior_types),1);
all_behavior_count  = zeros(numel(all_behavior_types),3);

for fn= 1:numel(play_behavior_struct_files)
    load([last_hmm_dir, '\',play_behavior_struct_files(fn).name]) 
    behavior_count = diag(play_behavior_struct.numbers)*play_behavior_struct.proportions;

    for bn = find(ismember(all_behavior_types,play_behavior_struct.behavior_types))'
        beh_index = ismember(play_behavior_struct.behavior_types, all_behavior_types(bn));
        all_numbers(bn) = all_numbers(bn)+play_behavior_struct.numbers(beh_index);

        all_behavior_count(bn,:) =  all_behavior_count(bn,:)+behavior_count(beh_index,:);
    end
end

behaviors2merge = {'Pounce_A','Pounce_B'};
re_name = {'Pounce'};
value = sum(all_behavior_count(ismember(all_behavior_types,behaviors2merge),:));
all_behavior_count(ismember(all_behavior_types,behaviors2merge),:) = repmat(value,numel(behaviors2merge), 1 );
value = sum(all_numbers(ismember(all_behavior_types,behaviors2merge),:));
all_numbers(ismember(all_behavior_types,behaviors2merge)) = value;


all_behavior_count(ismember(all_behavior_types,behaviors2merge(1)),:) = [];

all_numbers(ismember(all_behavior_types,behaviors2merge(1))) = [];
all_behavior_types(ismember(all_behavior_types,behaviors2merge(1))) = [];
all_behavior_types(ismember(all_behavior_types,behaviors2merge)) = re_name;

size(all_numbers)
size(all_behavior_count)
size(all_behavior_types)

behaviors2merge = {'Pounce_Ai','Pounce_Bi'};
re_name = {'PounceI'};
value = sum(all_behavior_count(ismember(all_behavior_types,behaviors2merge),:));
all_behavior_count(ismember(all_behavior_types,behaviors2merge),:) = repmat(value,numel(behaviors2merge), 1 );
value = sum(all_numbers(ismember(all_behavior_types,behaviors2merge),:));
all_numbers(ismember(all_behavior_types,behaviors2merge)) = value;

all_behavior_count(ismember(all_behavior_types,behaviors2merge(1)),:) = [];
all_numbers(ismember(all_behavior_types,behaviors2merge(1))) = [];
all_behavior_types(ismember(all_behavior_types,behaviors2merge(1))) = [];

all_behavior_types(ismember(all_behavior_types,behaviors2merge)) = re_name;


behaviors2delete = {'','Sniffing_C'};

all_behavior_count(ismember(all_behavior_types,behaviors2delete),:) = [];
all_numbers(ismember(all_behavior_types,behaviors2delete)) = [];
all_behavior_types(ismember(all_behavior_types,behaviors2delete)) = [];
%%  now plot barplot

all_behavior_proportions = diag(1./all_numbers)*all_behavior_count;

[~,bar_order] = sort(all_behavior_proportions(:,2), 'descend');
figure

toincludemove = ~ismember(all_behavior_types(bar_order), {'Partners session','Sniffing_C',''});
subplot(5,1,1)
bar(all_numbers( bar_order(toincludemove)))
xticks(1:numel(bar_order(toincludemove)))
xticklabels([])
axis tight


subplot(5,1,2:4)
bar(all_behavior_proportions(bar_order(toincludemove),[2,1,3]), 'stacked')
xticks(1:numel(bar_order(toincludemove)))

all_behavior_types_realNames = {'Unlabeled','Bite','Boxing','PlayfulApproach','chasing','nonPlayfulApproach','Escape','Evasion','Grooming',...
    'Partners session','Pin','PounceNeck','PounceNeckImmob','PounceBack','PounceBackImmob','Rearing','Scratch','Sniffing','SniffingCable'}; % ths is for 3 states
all_behavior_types_realNames = {'Unlabeled','Bite','Boxing','PlayfulApproach','Chasing','nonPlayfulApproach','Escape','Evasion','Grooming',...
    'Pin','PounceNeck','PounceNeckImmob','PounceBack','PounceBackImmob','Rearing','Scratch','Sniffing'};
% all_behavior_types_realNames= all_behavior_types
xticklabels(all_behavior_types(bar_order(toincludemove)))
axis tight

%% now save

   print(gcf,'-vector','-dsvg',[figure_dir,'\play axis hmm.svg'])
