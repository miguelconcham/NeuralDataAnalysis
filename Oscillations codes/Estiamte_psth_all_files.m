

npx_Raw_Data = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\NPX data\NPX raw data';
saving_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\Theta psth';
animal_list = dir(npx_Raw_Data);
animal_list(1:2) = [];
figure_3_new_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Codes\Figure codes\Figure 3 Updated';

animal_file_names =  cellfun(@(x) ['B', x],strsplit([animal_list.name], 'B'), 'UniformOutput',false)';
animal_file_names(1) = [];
% animal2exclude = {'B4D4 0826 Dual'};
animal2exclude = {''};
animal_list(ismember(animal_file_names,animal2exclude)) = [];
animal_names ={};
n_strctut = 1;

psth_structure = [];
wind_length     = 1;
wind_overlap    = .990;
min_separation = .200;
f               = .1:.1:6;
freq_pow_range  = [1 5];

%%
for fn = 1:numel(animal_list)

    if fn==1
        psth_structure = GENERATE_THETA_PSTH([npx_Raw_Data, '\', animal_list(fn).name],wind_length,wind_overlap,min_separation,f,freq_pow_range )
        n_strctut = n_strctut+numel(psth_structure);
        animal_names = [animal_names;[repmat(animal_list(fn).name,numel(psth_structure),1) num2cell(1:numel(psth_structure))']]
    else
        transt_psth = GENERATE_THETA_PSTH([npx_Raw_Data, '\', animal_list(fn).name],wind_length,wind_overlap,min_separation,f,freq_pow_range )

        for sub_j=1:numel(transt_psth)

            psth_structure(n_strctut) = transt_psth(sub_j);
            n_strctut = n_strctut+1;
        end
        animal_names = [animal_names;[repmat({animal_list(fn).name},numel(transt_psth),1) num2cell(1:numel(transt_psth))' ]]

    end


end
%% save if needed
disp('saving')
save([saving_folder,'\psth_structure_delta_updated.mat'],'psth_structure', '-v7.3');
save([saving_folder,'\animal_names_delta_updated.mat'],'animal_names');

%% load if needed
disp('loading')
% cd(saving_folder) = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\Theta psth';

load([saving_folder,'\psth_structure_delta_updated.mat'],'psth_structure');
load([saving_folder,'\animal_names_delta_updated.mat'],'animal_names');
disp('ready')
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
pre_time = [-.37 -.07];
pre_time_index = time<pre_time(2) & time>pre_time(1);
all_psth_onset                  = [];
all_psth_onset_behavior         = [];
all_psth_onset_only_playobut    = [];
all_psth_offset         = [];
all_psth_tw             = [];
all_psth_tw_3points     = [];
all_play_bouts          = [];
time_wrap_time          = [(baseline_index_time_wrap*bin_size) + wrap_range(1),linspace(0,1,psth_structure(1).n_bins_time_wrap),1 + (1:round((abs(wrap_range(2))/bin_size)))*bin_size];
time_wrap_3_points      = [(baseline_index_time_wrap*bin_size) + wrap_range(1),linspace(0,1-1/psth_structure(1).n_bins_time_wrap,psth_structure(1).n_bins_time_wrap), ...
   linspace(1,2-1/psth_structure(1).n_bins_time_wrap,psth_structure(1).n_bins_time_wrap),2 + (1:round((abs(wrap_range(2))/bin_size)))*bin_size];
partner_number = [];
session_index = [];
animal_index = [];
baseline_values = [];
play_per_partner = [];
play_per_partner_per_session = [];
baseline_values_per_session = [];
all_electrodes = [];

animal_index_per_session = []
for j=1:numel(psth_structure)

    if contains(animal_names{j},animal_label)
            
        animal_num      = find(cell2mat(cellfun(@(x) contains(animal_names{j,1},x), animal_label, 'UniformOutput',false)));  
        electrode_num   =animal_names{j,2};
        this_animal_playbouts = psth_structure(j).play_bouts_table;
        this_animal_lengths = diff(this_animal_playbouts');
        all_play_bouts = [all_play_bouts;this_animal_playbouts];   
        partner_sessions = psth_structure(j).Behavior(ismember(psth_structure(j).Behavior.Animal,'Session_structure'),:);   
        partner_sessions(ismember(partner_sessions.Type, 'Tickling'),:) = [];
        [~, loc]      = max(this_animal_playbouts(:,1)'>=partner_sessions.Start & this_animal_playbouts(:,2)'<partner_sessions.End,[],1);
        partner_number = [partner_number;loc'];
            
        
       animal_index_per_session= [animal_index_per_session;animal_num];

    


      this_psth_onset         = psth_structure(j).play_bout_onset;
      this_psth_onset_onlypb  = this_psth_onset;
      animal_index = [animal_index;repmat(animal_num,size(this_psth_onset,1),1)];
      session_index = [session_index;repmat(j,size(this_psth_onset,1),1)];
      
      for trial=1:size(this_psth_onset,1)
          this_psth_onset(trial,:) = ( this_psth_onset(trial,:) - mean( this_psth_onset(trial,baseline_index)))/std( this_psth_onset(trial,baseline_index));
          this_psth_onset(trial,:) = movmean(this_psth_onset(trial,:), smooth_wind);
          % this_psth_onset_onlypb(trial,:) = this_psth_onset(trial,:);
          this_psth_onset(trial,time> this_animal_lengths(trial)) = NaN;
      end
                 baseline_this_session=mean(this_psth_onset(:,pre_time_index),2);
                baseline_values = [baseline_values;baseline_this_session];

                  play_per_partner_this_animal = nan(1,3);
          baseline_values_this_session = nan(1,3); 
          for pn=1:size(partner_sessions,1)
              session_length = partner_sessions.End(pn) - partner_sessions.Start(pn);
              play_per_partner_this_animal(pn) = sum(this_animal_playbouts(loc==pn,2)-this_animal_playbouts(loc==pn,1))/session_length;

              baseline_values_this_session(pn)  = mean(baseline_this_session(loc==pn));
          end
        play_per_partner = [play_per_partner;repmat(play_per_partner_this_animal,size(this_psth_onset,1),1)];
        play_per_partner_per_session = [play_per_partner_per_session;play_per_partner_this_animal];
        baseline_values_per_session =[baseline_values_per_session;baseline_values_this_session] ;

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
             this_psth_ab(trial,time> this_animal_lengths(trial)) = NaN;
        end
        all_psth_onset_behavior = [all_psth_onset_behavior; this_psth_ab];
    all_electrodes = [all_electrodes;electrode_num];
    end
end

play_bout_length = diff(all_play_bouts')';


[sorted_play_bout_length, order] = sort(play_bout_length);

%%

X_lim = [-1 2];
y_lim = [-1 3];
alpha= 0.01;

figure
staked_means = [];
time2use = time;
data_for_paired = [];
for an= 1:numel(animal_label)

    [sorted_play_bout_length, order] = sort(play_bout_length(animal_index==an,:));
    subplot(5,numel(animal_label),(1:numel(animal_label):2*numel(animal_label)) + an-1)
    array = all_psth_onset(animal_index==an,:);
    parthers = partner_number(animal_index==an);
    play_per_partner_this_session  = play_per_partner(animal_index==an,:);
    this_baseline_values = baseline_values(animal_index==an);
    imagesc(time2use,1:numel(sorted_play_bout_length),array(order,:) )
    xlim(X_lim)
    clim([-2 2])
    axis xy
    hold on
    imagesc_y_lim = ylim;
    plot([0 0], imagesc_y_lim, 'w')
    plot(sorted_play_bout_length, 1:numel(sorted_play_bout_length), 'w')


    subplot(5,numel(animal_label),((2*numel(animal_label) + 1):numel(animal_label):4*numel(animal_label)) + an-1)

    [~, ~, ci]  = ttest(array(:,:));
    no_nan = ~any(isnan(ci));
    fill([time2use(no_nan) fliplr(time2use(no_nan))], [ci(1,no_nan) fliplr(ci(2,no_nan))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time2use,mean(array(:,:), 'omitmissing'), 'k')

    [~, ~, ci]  = ttest(array(parthers==2,:));
    no_nan = ~any(isnan(ci));
    fill([time2use(no_nan) fliplr(time2use(no_nan))], [ci(1,no_nan) fliplr(ci(2,no_nan))], 'r', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time2use,mean(array(parthers==2,:), 'omitmissing'), 'r')
    xlim(X_lim)
    ylim(y_lim)

      subplot(5,numel(animal_label),(5*(numel(animal_label)-1)) + an-1)
    expected_orders = {[1 2],[2 1]};
    this_baseline_values = zscore(this_baseline_values);
      swarmchart(cell2mat(expected_orders((play_per_partner_this_session(:,2)>play_per_partner_this_session(:,1) )+1)'),this_baseline_values, 'k.')
      hold on
      play_lengths= mean(play_per_partner_this_session);
      % play_lengths = play_lengths/max(play_lengths);
      this_staked_means = nan(numel(play_lengths),1);
      for j=1:3
          index = partner_number==j;
          if sum(index)>0
              this_staked_means(j) = mean(this_baseline_values(parthers==j));
              plot([.5 1.5] + j-1,[1 1]*mean(this_baseline_values(parthers==j)), 'r')
          end
      end

      staked_means =[staked_means; [play_lengths' this_staked_means,ones(numel(this_staked_means),1)*an (1:3)']];
      data_for_paired = [data_for_paired;play_lengths];
end

%% plot partne reffect

figure


 


subplot(1,3,1)
x_rand = (rand(sum(all_electrodes==1),2)-.5)/2 + ones(sum(all_electrodes==1),2)*diag([1 2]);
plot(x_rand', play_per_partner_per_session(all_electrodes==1, 1:2)', ':k')
hold on
    plot(x_rand', play_per_partner_per_session(all_electrodes==1, 1:2)', '.k', 'MarkerSize', 15)

 p = signrank(play_per_partner_per_session(all_electrodes==1, 1), play_per_partner_per_session(all_electrodes==1, 2));
title(p)

reduced = play_per_partner_per_session(:,2)<play_per_partner_per_session(:,1);
axis square
subplot(1,3,2)



table_data = [play_per_partner_per_session(all_electrodes==1,1)-play_per_partner_per_session(all_electrodes==1,2),baseline_values_per_session(all_electrodes==1,1)-baseline_values_per_session(all_electrodes==1,2), animal_index_per_session(all_electrodes==1)]
table_data = array2table(table_data);
table_data.Properties.VariableNames = {'x','y','Subject'};
lme = fitlme(table_data, 'y ~ 1 + x+ (1|Subject)');

% Fit model
lme = fitlme(table_data, 'y ~ 1 + x + (1|Subject)');

% Create x-grid for smooth prediction
xvals = linspace(min(table_data.x), max(table_data.x), 100)';
tbl_pred = table(xvals, 'VariableNames', {'x'});

% Get predicted mean and confidence intervals
tbl_pred = table(xvals, repmat(table_data.Subject(1), numel(xvals), 1), ...
    'VariableNames', {'x','Subject'});
[yhat, yCI] = predict(lme, tbl_pred, 'Conditional', false);% 'Conditional', false => marginal over random effects (population-level line)

% Plot data
 hold on
scatter(table_data.x, table_data.y, 20, 'k', 'filled', 'MarkerFaceAlpha', 0.3)

% Plot confidence band (as shaded area)
fill([xvals; flipud(xvals)], ...
     [yCI(:,1); flipud(yCI(:,2))], ...
     [0.8 0.8 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3);

% Plot fitted line
plot(xvals, yhat, 'b', 'LineWidth', 2);
xlabel('Play change (Partner 1- Partner 2)')
ylabel('Delta power change (Partner 1- Partner 2)')
title( num2str([lme.anova.FStat lme.anova.pValue]))
axis square

subplot(1,3,3)

no_nan = ~any(isnan(staked_means(:,1:2)),2)
plot(staked_means(:,1), staked_means(:,2), '.k')
hold on
sesions = unique(staked_means(:,3))
for ns=1:numel(sesions)
    x = staked_means(staked_means(:,3) ==sesions(ns),1);
    y = staked_means(staked_means(:,3) ==sesions(ns),2);
    parter_number = 1:3;

    [~, correct_order] =sort(x);
    parter_number = parter_number(correct_order);
    if ns==5
         plot( x(correct_order),y(correct_order) ,'.r')
       plot( x(correct_order),y(correct_order) ,'r')
       for pn=1:numel(parter_number)
       text(x(correct_order(pn)),y(correct_order(pn)), ['Pt #',num2str(parter_number(pn))], 'Color', 'r')
       end
    else
      plot( x(correct_order),y(correct_order) ,'.k')
       plot( x(correct_order),y(correct_order) ,'k')
       for pn=1:numel(parter_number)
       text(x(correct_order(pn)),y(correct_order(pn)), ['Pt #',num2str(parter_number(pn))])
       end
    end
end

[c,p] = corr(staked_means(no_nan,1), staked_means(no_nan,2), 'Type','Spearman')
title([c p])


% p = ranksum(baseline_values(partner_number==1), baseline_values(partner_number==2))
title(p)
axis square
%%

staked_mean_responses = cell(3,1);
min_length = .0;
pn=1;
for partner_number_list = {1, 2, [1 2 3]};
    figure
    staked_mean_responses{pn} = [];
for an= 1:numel(animal_label)
    animal_bool = animal_index==an & ismember(partner_number, partner_number_list{1});
    length_bool = play_bout_length>min_length;
    [sorted_play_bout_length, order] = sort(play_bout_length(animal_bool & length_bool,:));
    subplot(5,numel(animal_label),(1:numel(animal_label):2*numel(animal_label)) + an-1)

    array = all_psth_onset(animal_bool & length_bool,:);
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
     staked_mean_responses{pn} = [ staked_mean_responses{pn};mean(array, 'omitmissing')]
    xlim(X_lim)
end
pn =pn+1;
end



%% onst to behavior

figure
min_length = .0;

for an= 1:numel(animal_label)
    animal_bool = animal_index==an;
    length_bool = play_bout_length>min_length;
    [sorted_play_bout_length, order] = sort(play_bout_length(animal_bool & length_bool,:));
    subplot(5,numel(animal_label),(1:numel(animal_label):2*numel(animal_label)) + an-1)

    array = all_psth_onset_behavior(animal_bool & length_bool,:);
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
    fill([time fliplr(time)], [ci(1,:) fliplr(ci(2,:))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time,mean(array, 'omitmissing'), 'k')
    xlim(X_lim)
end

%% timewrpa playbout and play behavior

         
X_lim = [-1 2]

figure
min_length = .0;

for an= 1:numel(animal_label)
    animal_bool = animal_index==an;
    length_bool = play_bout_length>min_length;
    [sorted_play_bout_length, order] = sort(play_bout_length(animal_bool & length_bool,:));
    subplot(5,numel(animal_label),(1:numel(animal_label):2*numel(animal_label)) + an-1)

    array = all_psth_tw_3points(animal_bool & length_bool,:);
    imagesc(time_wrap_3_points,1:numel(sorted_play_bout_length),array(order,:) )
    xlim(X_lim)
    clim([-2 2])
    axis xy
    hold on
    plot([0 0],[1 numel(sorted_play_bout_length)], 'w')
    plot(sorted_play_bout_length,1:numel(sorted_play_bout_length), 'w')
    title(animal_label{an})

    subplot(5,numel(animal_label),((2*numel(animal_label) + 1):numel(animal_label):5*numel(animal_label)) + an-1)

    [~, ~, ci]  = ttest(array);
    fill([time_wrap_3_points fliplr(time_wrap_3_points)], [ci(1,:) fliplr(ci(2,:))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time_wrap_3_points,mean(array, 'omitmissing'), 'k')
    xlim(X_lim)
end


%%
min_length = 0;
length_bool = play_bout_length>min_length;
figure
    array = all_psth_tw_3points( length_bool,:);

    [~, ~, ci]  = ttest(array);
    fill([time_wrap_3_points fliplr(time_wrap_3_points)], [ci(1,:) fliplr(ci(2,:))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time_wrap_3_points,mean(array, 'omitmissing'), 'k')
    xlim(X_lim)



%%

figure
hold on
array = all_psth_onset;

 [~, ~, ci]  = ttest(array);
 fill([time fliplr(time)], [ci(1,:) fliplr(ci(2,:))], 'r', 'FaceAlpha',.25, 'EdgeColor','none')
plot(time,mean(array, 'omitmissing'), 'r')
hold on

array = all_psth_onset;
 [~, ~, ci]  = ttest(array);
 fill([time fliplr(time)], [ci(1,:) fliplr(ci(2,:))], 'b', 'FaceAlpha',.25, 'EdgeColor','none')
plot(time,mean(array, 'omitmissing'), 'b')
plot(time, mean(all_psth_onset), 'b')
 xlim(X_lim)
%% estimating mixed model per time

length_limit =4;
array2use = all_psth_onset;
partner_names = {'Partner1', 'Partner2', 'Partner3'};
partner_matrix = repmat(partner_number,1, size(array2use,2));
indextinclude = play_bout_length>=length_limit;
% power = all_psth_onset_behavior(indextinclude,:);
power = array2use(indextinclude,:);
zscore_limit = 10;
subject_idx =animal_index(indextinclude);
time_range = [-5 5];
limted_time = time;
% limted_time =time_wrap_3_points;
valid_bins = find(limted_time >= time_range(1) & limted_time <= time_range(2));
power = power(:,valid_bins);
limted_time = limted_time(valid_bins);

[nTrials, nTime] = size(power);

% Preallocate
est             = nan(nTime,1);
se              = nan(nTime,1);
pvals           = nan(nTime,1);
d               = nan(nTime,1);
ci              = nan(nTime,2);
partner_est     = nan(nTime,1);
partner_pvals   = nan(nTime,1);
partner_ci      = nan(nTime,2);
partner_se      = nan(nTime,1);
partner_d       = nan(nTime,1);



% Reshape to long format
[trial_idx, time_idx] = ndgrid(1:nTrials,1:nTime);
tbl = length_table;
tbl.Power = power(:);
tbl.Subject = categorical(subject_idx(trial_idx(:)));
tbl.Partner = categorical(partner_names(partner_number(trial_idx(:))))';
time_matrix = repmat(limted_time, nTrials, 1);
tbl.Time = time_matrix(:);
length_matrix = repmat(play_bout_length(indextinclude),1,numel(limted_time));
tbl.length = length_matrix(:);

tbl(abs(tbl.Power)>zscore_limit,:) = []; %remove outliers if needed
tbl(isnan(tbl.Power),:) = [];
tbl(tbl.Partner==categorical(partner_names(3)),:) = [];  %remove sesions with 3 partners

subject_list = unique(tbl.Subject);
animal_counts_per_bin = nan(nTime,numel(subject_list));
re = nan(nTime,numel(subject_list));
partner_intercept_name = cell(nTime,1);

% Loop over bins
for i = 1:nTime
    tbl_t = tbl(tbl.Time == limted_time(i) & tbl.Time<tbl.length,:);
   [animal_this_bin, id] = groupcounts(tbl_t.Subject);
   animal_counts_per_bin(i,ismember(subject_list,id )) = animal_this_bin;
   if min(animal_this_bin)>10
       tbl_t.Partner = removecats(tbl_t.Partner);
       lme = fitlme(tbl_t,'Power ~ 1 + Partner + (1|Subject)');
       re(i,:) = randomEffects(lme);

       est(i) = lme.Coefficients.Estimate(1);
       partner_est(i) = lme.Coefficients.Estimate(2);
       se(i)  = lme.Coefficients.SE(1);
       partner_se(i) = lme.Coefficients.SE(2);
       pvals(i) = lme.Coefficients.pValue(1);
       partner_pvals(i) = lme.Coefficients.pValue(2);
       lme.Coefficients.Name(2)
       ci_t = coefCI(lme);
       ci(i,:) = ci_t(1,:);
       partner_ci(i,:) = ci_t(2,:);

       % Cohen's d-like standardized effect size
       sd_within = std(tbl_t.Power, 'omitmissing');
       d(i) = est(i) / sd_within;
       partner_d(i) = partner_est(i) / sd_within;
   end
end

% Multiple comparison correction (FDR)
pvals_fdr = mafdr(pvals,'BHFDR',true);

% Package results
results.time        = limted_time(:);
results.est         = est;
results.se          = se;
results.ci          = ci;
results.d           = d;
results.pvals       = pvals;
results.pvals_fdr   = pvals_fdr;

results.partner_est         = partner_est;
results.partner_se          = partner_se;
results.partner_ci          = partner_ci;
results.partner_d           = partner_d;
results.partner_pvals       = partner_pvals;
results.partner_pvals_fdr   = mafdr(partner_pvals,'BHFDR',true);;


results.zscore_limit = zscore_limit;
results.time_range = time_range;
play_song([],[],[])
%% save if needed

save([saving_folder,'\results_play_bout_PBonly_zscore4_updated_longer_than_4.mat'],'results');

%% load if needed
% load([saving_folder,'\results_play_bout.mat'],'results');
% load([saving_folder,'\results_play_bout_PBonly_zscore4_updated.mat'],'results');   %     here we use all play bouts
  

load([saving_folder,'\results_play_bout_PBonly_zscore4_updated_longer_than_4.mat'],'results'); %     here we use only playbouts longer than 4 saeconds 

%%
x_lim = [-1 2];
alpha = 0.05;


limited_time = results.time;
est = results.est;
ci = results.ci;
% pvals_fdr = results.pvals_fdr;
pvals_fdr = results.pvals;
d = results.d;


% % 
% est = results.partner_est;
% ci = results.partner_ci ;      
% d = results.partner_d  ;       
% pvals_fdr = results.partner_pvals      ;

figure;
subplot(2,1,1); hold on;

% Shaded CI
what_to_plot = (staked_mean_responses{2}-staked_mean_responses{1})';
what_to_plot = staked_mean_responses{3};
plot(time,what_to_plot, 'k')
no_nan = ~any(isnan(ci),2)
fill([limited_time(no_nan); flipud(limited_time(no_nan))], [ci(no_nan,1); flipud(ci(no_nan,2))], ...
    [0.8 0.8 1], 'EdgeColor','none','FaceAlpha',0.4);
plot(limited_time, est, 'b','LineWidth',2);

% Mark significant bins
sig_idx = pvals_fdr < alpha;
plot(limited_time(sig_idx), est(sig_idx), 'r*','MarkerSize',6);

ylabel('Mean Power');
title('Mixed-Effects Power (CI + FDR-corrected sig)');
grid on;
xlim(x_lim)

subplot(2,1,2);
plot(limited_time, d, 'k','LineWidth',2);
ylabel('Effect size (Cohen''s d)');
xlabel('Time (s)');
grid on;
xlim(x_lim)

%%

alpha = 0.05;
[cluster_pvals, clusters, cluster_stats, perm_max_stats] = cluster_perm_test(array(hmm_type==1,:), array(hmm_type==0,:), alpha, 500);

save([saving_folder,'\cluster_pvals.mat'],'cluster_pvals','clusters','cluster_stats','perm_max_stats', 'array', 'hmm_type','alpha', 'time')

%% delta onset per session

all_animals_delta_onset = [];
for j=1:numel(psth_structure)

    this_session = session_index==j;

all_animals_delta_onset = [all_animals_delta_onset;mean(all_psth_onset(this_session,:), 'omitmissing')];
end



%%

figure
plot(time,all_animals_delta_onset, 'k:')
hold on
[~, ~, ci] = ttest(all_animals_delta_onset);

plot(time, mean(all_animals_delta_onset), 'k')
xlim([-3 4])