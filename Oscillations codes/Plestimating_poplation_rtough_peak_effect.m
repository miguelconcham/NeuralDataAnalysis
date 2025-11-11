%% fnd session combinations
an = 3;

area_index      = ismember(all_neurons_TD.area, area_list{an});
angletype_peak  = ~(all_neurons_TD.(freq2use).PreferedAngle>=pi/2 | all_neurons_TD.(freq2use).PreferedAngle<-pi/2);
no_nan          = ~isnan(all_neurons_TD.Inhibited);
response_type = 'All';
    histogram_edge = -2:0.25:8;
alpha_level = 0.01;
non_entrained_lvl =0.1;
if strcmp(response_type, 'All')
    response_index = true(size(all_neurons_TD, 1),1);
else
    response_index  = all_neurons_TD.(response_type)==1;
end

entreined       = all_neurons_TD.(freq2use).PPCPval<=alpha_level ;
not_entrained   = all_neurons_TD.(freq2use).PPCPval>non_entrained_lvl ;

peak_index          =   area_index & no_nan & response_index & entreined & angletype_peak  ;
trough_index        =  area_index & no_nan & response_index & entreined & ~angletype_peak ;
nonentrained_index  =  area_index & no_nan & response_index & not_entrained ;

session_name = all_neurons_TD.session,session_list = unique(session_name);

unique_sessions = unique(session_name);
n = length(unique_sessions);

% Initialize cell for valid session combinations
valid_combos = [];
amount_per_type = [];


indexes = false(numel(psth_list),1);
 selected_behaviors              =  [1 2 3 4 5 6 7 8 9 14 15 18 19 20 21 24 25];

% Check all possible combinations of sessions (up to all n sessions)
for k = 1:n-1
    combos = nchoosek(1:n, k);  % all combinations of size k
    for j = 1:size(combos,1)
        idx = ismember(session_name, unique_sessions(combos(j,:)));
        
        % Check if all three clusters appear at least once
        if any(peak_index(idx)) && any(trough_index(idx)) && any(nonentrained_index(idx))
            % Save the session names
            this_combos = false(1,n);
            this_combos(combos(j,:)) = true;
            valid_combos =[valid_combos; this_combos]; %#ok<AGROW>
            amount_per_type = [amount_per_type; sum([peak_index(idx) , trough_index(idx)  nonentrained_index(idx)])];
        end
    end
end
a = min(amount_per_type,[],2);
accepted_combinations = find(a>5 & sum(valid_combos,2)<8);
numel(accepted_combinations)
%% Estimating population respones (logit) for all behaviors and all areas
% alpha_level  = 0.05;
% non_entrained_lvl = 0.1;
warped_time     = (((1:60)/20)*5) - 5 ;
smooth_window = 5;

baseline  = [-Inf 0];


freq2use        = 'DeltaEntireSession'; %options:   ThetaEntireSession DeltaEntireSession
area_list       = {'SupCol' 'DLPAG'	'LPAG'	'VLPAG' 'DR' };
plot_angle      = true;


n_perm = 10000;

all_indexes = [];
psth_names = {'ChaseSelf','ChaseOther','PounceSelf','PounceOther','PounceWISelf','PounceWIOther','PlaySelf','PlayOther'};
psth_list = [{'PsthWarped'}';psth_map(:,1)];
psth_list([  4 5]) = [];


response_time = [0 5];


pop_activation              = nan(numel(psth_list),3);
%%
an =3;
comb_results              = nan(numel(accepted_combinations),numel(psth_list),4);
com_psth_results         = nan(numel(accepted_combinations),numel(psth_list),3,size(warped_time,2));
com_psth_results_pctg    = nan(numel(accepted_combinations),numel(psth_list),3,size(warped_time,2));
          prefix = 'Boostrap iteration # ';      % fixed text
        fprintf('\n%s1', prefix);             % print first number
for sn=1:numel(accepted_combinations)

      

            if sn>1
             prev_num_digits = floor(log10(sn-1)) + 1;

            % backspace previous digits
            for k = 1:prev_num_digits
                fprintf('\b');
            end

            % print current number
            fprintf('%d', sn);
            pause(0.1);
            end

    
    session_index =ismember(all_neurons_TD.session,unique_sessions(valid_combos(accepted_combinations(sn),:)==1) );
    psth_n = 1;
    for psth2use_cell = psth_list'
        psth2use =  psth2use_cell{1};
        if strcmp(psth2use, 'PsthOnlyPB')
            time2use = mig_edges_centers;
        elseif strcmp(psth2use, 'PsthOnset') || strcmp(psth2use, 'PsthOffset')
            time2use= non_wraped_time;
        else

            time2use = warped_time;
        end
        baseline_index      = time2use>=baseline(1) & time2use<=baseline(2);
        response_time_index = time2use>=response_time(1) & time2use<=response_time(2);






        area_index      = ismember(all_neurons_TD.area, area_list{an});
        angletype_peak  = ~(all_neurons_TD.(freq2use).PreferedAngle>=pi/2 | all_neurons_TD.(freq2use).PreferedAngle<-pi/2);
        no_nan          = ~isnan(all_neurons_TD.Inhibited);
        if strcmp(response_type, 'All')
            response_index = true(size(all_neurons_TD, 1),1);
        else
            response_index  = all_neurons_TD.(response_type)==1;
        end

        entreined       = all_neurons_TD.(freq2use).PPCPval<=alpha_level ;
        not_entrained   = all_neurons_TD.(freq2use).PPCPval>non_entrained_lvl ;

        peak_index          = session_index &  area_index & no_nan & response_index & entreined & angletype_peak  ;
        trough_index        = session_index &  area_index & no_nan & response_index & entreined & ~angletype_peak ;
        nonentrained_index  = session_index &  area_index & no_nan & response_index & not_entrained ;



        peak_psth                           = all_neurons_TD.(psth2use)(peak_index, :);
        for j=1:size(peak_psth,1)
            peak_psth(j,:) = smooth( peak_psth(j,:),smooth_window);
            if  std( peak_psth(j,baseline_index), 'omitmissing')>0.01
                peak_psth(j,:) = ( peak_psth(j,:) - mean( peak_psth(j,baseline_index), 'omitmissing'))/ std( peak_psth(j,baseline_index), 'omitmissing');
            else
                peak_psth(j,:) = ( peak_psth(j,:) - mean( peak_psth(j,:), 'omitmissing'))/ std( peak_psth(j,:), 'omitmissing');
            end
        end
        trough_psth  =  all_neurons_TD.(psth2use)(trough_index, :);
        for j=1:size(trough_psth,1)
            trough_psth(j,:) = smooth( trough_psth(j,:),smooth_window);
            if  std( trough_psth(j,baseline_index), 'omitmissing')>0.01
                trough_psth(j,:) = ( trough_psth(j,:) - mean( trough_psth(j,baseline_index), 'omitmissing'))/ std( trough_psth(j,baseline_index), 'omitmissing');
            else
                trough_psth(j,:) = ( trough_psth(j,:) - mean( trough_psth(j,:), 'omitmissing'))/ std( trough_psth(j,:), 'omitmissing');
            end
        end
        nonentrained_psth  =  all_neurons_TD.(psth2use)(nonentrained_index, :);
        for j=1:size(nonentrained_psth,1)
            nonentrained_psth(j,:) = smooth( nonentrained_psth(j,:),smooth_window);
            if  std( nonentrained_psth(j,baseline_index), 'omitmissing')>0.01
                nonentrained_psth(j,:) = ( nonentrained_psth(j,:) - mean( nonentrained_psth(j,baseline_index), 'omitmissing'))/ std( nonentrained_psth(j,baseline_index), 'omitmissing');
            else
                nonentrained_psth(j,:) = ( nonentrained_psth(j,:) - mean( nonentrained_psth(j,:), 'omitmissing'))/ std( nonentrained_psth(j,:), 'omitmissing');
            end
        end
        all_psth = all_neurons_TD.(psth2use)(area_index & no_nan & response_index, :);
        for j=1:size(all_psth,1)
            all_psth(j,:) = smooth( all_psth(j,:),smooth_window);
            if  std( all_psth(j,baseline_index), 'omitmissing')>0.01
                all_psth(j,:) = ( all_psth(j,:) - mean( all_psth(j,baseline_index), 'omitmissing'))/ std( all_psth(j,baseline_index), 'omitmissing');
            else
                all_psth(j,:) = ( all_psth(j,:) - mean( all_psth(j,:), 'omitmissing'))/ std( all_psth(j,:), 'omitmissing');
            end
        end






        null_psth_peak = nan(n_perm, size(peak_psth,2));
        for pn = 1:n_perm

            sub_selection = all_psth(randperm(size(all_psth,1),size(peak_psth,1)),:);

            null_psth_peak(pn,:) = median(sub_selection, 'omitmissing');
        end
        % peak_pctl_activation = 100*mean(null_psth>mean(peak_psth, 'omitmissing'));
        peak_pctl_activation = mean(null_psth_peak<median(peak_psth, 'omitmissing'));
        peak_pctl_activation(peak_pctl_activation==0) = 1/n_perm;
        peak_pctl_activation(peak_pctl_activation==1) = (n_perm-1)/n_perm;
        peak_pctl_activation = log(peak_pctl_activation./(1-peak_pctl_activation));
      
        null_psth_trough = nan(n_perm, size(trough_psth,2));
        for pn = 1:n_perm

            sub_selection = all_psth(randperm(size(all_psth,1),size(trough_psth,1)),:);

            null_psth_trough(pn,:) = median(sub_selection, 'omitmissing');
        end
        % trough_pctl_activation = 100*mean(null_psth>mean(trough_psth, 'omitmissing'));
        trough_pctl_activation = mean(null_psth_trough<median(trough_psth, 'omitmissing'));

        trough_pctl_activation(trough_pctl_activation==0) = 1/n_perm;
        trough_pctl_activation(trough_pctl_activation==1) = (n_perm-1)/n_perm;
        trough_pctl_activation = log(trough_pctl_activation./(1-trough_pctl_activation));
     

        null_psth_nonentrained = nan(n_perm, size(nonentrained_psth,2));
        for pn = 1:n_perm

            sub_selection = all_psth(randperm(size(all_psth,1),size(nonentrained_psth,1)),:);

            null_psth_nonentrained(pn,:) = median(sub_selection, 'omitmissing');
        end
        % nonentrained_pctl_activation = 100*median(null_psth>median(nonentrained_psth, 'omitmissing'));
        nonentrained_pctl_activation = mean(null_psth_nonentrained<median(nonentrained_psth, 'omitmissing'));
        nonentrained_pctl_activation(nonentrained_pctl_activation==0) = 1/n_perm;
        nonentrained_pctl_activation(nonentrained_pctl_activation==1) = (n_perm-1)/n_perm;
        nonentrained_pctl_activation = log(nonentrained_pctl_activation./(1-nonentrained_pctl_activation));
       


        com_psth_results(sn,psth_n,1,:) =   peak_pctl_activation;
        com_psth_results(sn,psth_n,2,:) =   trough_pctl_activation;
        com_psth_results(sn,psth_n,3,:) =   nonentrained_pctl_activation;

        com_psth_results(sn,psth_n,1,:) =   peak_pctl_activation;
        com_psth_results(sn,psth_n,2,:) =   trough_pctl_activation;
        com_psth_results(sn,psth_n,3,:) =   nonentrained_pctl_activation;

        com_psth_results_pctg(sn,psth_n,1,:)  =sum(peak_psth)./sum(all_psth);
        com_psth_results_pctg(sn,psth_n,2,:)  =sum(trough_psth)./sum(all_psth);
        com_psth_results_pctg(sn,psth_n,3,:)  =sum(nonentrained_psth)./sum(all_psth);

        peak_pop            = median(peak_pctl_activation(response_time_index));
        trough_pop          = median(trough_pctl_activation(response_time_index));
        nonentrained_pop    = median(nonentrained_pctl_activation(response_time_index));

        pop_activation(psth_n,:) = [peak_pop trough_pop nonentrained_pop];

        psth_n = psth_n+1;
    end


    comb_results(sn,:,:) = [pop_activation, indexes*0+sn];
end

%%
labels = {'PlayBout'	'Self'	'Other'	'CHSelf'	'CHOther'	'POASelf'	'POAOther'	'POBSelf'	'POBOther'	'PWIASelf'	'PWIAOther'	'PWIBSelf'	'PWIBOther'	'EVSelf'	'EVOther'	'RESelf'	'REOther'	'ESSelf'	'ESOther'	'CDSelf'	'CDOther'	'SNSelf'	'SNOther'	'CBSelf'	'CBOther'	'GRSelf'	'GROther'	'Partner1Self'	'Partner1Other'	'Partner2Self'	'Partner2Other'};

saving_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\Theta psth';

save([saving_folder,'\all_neurons_comb_results.mat'],'comb_results', 'unique_sessions', 'valid_combos' ,'accepted_combinations','amount_per_type','com_psth_results','time2use', 'labels','com_psth_results_pctg');
%%
saving_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\Theta psth';

load([saving_folder,'\all_neurons_comb_results.mat']);

%%
min_per_type_accepted = min(amount_per_type(accepted_combinations,:),[],2)>=10;
selected_behaviors              =  [1 2 3 4 5 6 7 8 9 14 15 18 19 20 21 24 25];

selected_behaviors = [26 27 28 29]
selected_behaviors = 1:numel(psth_list);
indexes = false(numel(psth_list),1);
indexes(selected_behaviors)=true;

figure
subplot(2,3,1)
all_corr12 = nan(size(comb_results,1),2);
for sn=1:size(comb_results,1)
    [c,p]=corr(squeeze(comb_results(sn,:,1))',squeeze(comb_results(sn,:,2))');
    all_corr12(sn,:) = [c,p];
end

swarmchart(all_corr12(min_per_type_accepted,1)*0, all_corr12(min_per_type_accepted,1), '.')
ylim([-1 1])
title(['Peak and Trough', num2str(signrank(all_corr12(min_per_type_accepted,1)))])

subplot(2,3,2)
all_corr13 = nan(size(comb_results,1),2);
for sn=1:size(comb_results,1)
    [c,p]=corr(squeeze(comb_results(sn,:,1))',squeeze(comb_results(sn,:,3))');
    all_corr13(sn,:) = [c,p];
end

swarmchart(all_corr13(min_per_type_accepted,1)*0, all_corr13(min_per_type_accepted,1), '.')
ylim([-1 1])
title(signrank(all_corr13(min_per_type_accepted,1)))
title(['Peak and Non Modulaed', num2str(signrank(all_corr13(:,1)))])

subplot(2,3,3)
all_corr23 = nan(size(comb_results,1),2);
for sn=1:size(comb_results,1)
    [c,p]=corr(squeeze(comb_results(sn,:,2))',squeeze(comb_results(sn,:,3))');
    all_corr23(sn,:) = [c,p];
end

swarmchart(all_corr23(min_per_type_accepted,1)*0, all_corr23(min_per_type_accepted,1), '.')
ylim([-1 1])
title(signrank(all_corr23(:,1)))
title(['Trough and Non Modulaed', num2str(signrank(all_corr23(min_per_type_accepted,1)))])

subplot(2,2,3)
hold on

plot(all_corr12(min_per_type_accepted,1), all_corr13(min_per_type_accepted,1), '.')

ylim([-1 1])
xlim([-1 1])
plot([-1 1],[-1 1], 'r')
title(signrank(all_corr12(min_per_type_accepted,1)-all_corr13(min_per_type_accepted,1)))

subplot(2,2,4)

plot(all_corr12(min_per_type_accepted,1), all_corr23(min_per_type_accepted,1), '.')
hold on

ylim([-1 1])
xlim([-1 1])
plot([-1 1],[-1 1], 'r')
title(signrank(all_corr12(min_per_type_accepted,1)-all_corr23(min_per_type_accepted,1)))

%%
figure
face_alpha = 0.2;;
what2plot = find(indexes);
types2plot = [ 1 2];
for j=1:numel(what2plot)
    subplot(5,6,j)
    com_psth_results_means = squeeze(mean(com_psth_results(min_per_type_accepted,what2plot(j),types2plot(1),:),2, 'omitmissing'));



 pctl2plot =  prctile(com_psth_results_means,[5 95]);
    fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'r', 'FaceAlpha',face_alpha, 'EdgeColor','none')
    hold on
    pctl2plot =  prctile(com_psth_results_means,[10 90]);
    fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'r', 'FaceAlpha',face_alpha, 'EdgeColor','none')
    % pctl2plot =  prctile(com_psth_results_means,[1 99]);
    % fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'r', 'FaceAlpha',face_alpha, 'EdgeColor','none')
    hold on
    plot(time2use, median(com_psth_results_means, 'omitmissing'), 'Color', 'r')
% 

com_psth_results_means = squeeze(mean(com_psth_results(min_per_type_accepted,what2plot(j),types2plot(2),:),2, 'omitmissing'));
pctl2plot =  prctile(com_psth_results_means,[5 95]);
    fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'b', 'FaceAlpha',face_alpha, 'EdgeColor','none')
    hold on
    pctl2plot =  prctile(com_psth_results_means,[10 90]);
    fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'b', 'FaceAlpha',face_alpha, 'EdgeColor','none')
    % pctl2plot =  prctile(com_psth_results_means,[1 99]);
    % fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'b', 'FaceAlpha',face_alpha, 'EdgeColor','none')
    hold on
    plot(time2use, median(com_psth_results_means, 'omitmissing'), 'Color', 'b')
% 
%  pctl2plot =  prctile(com_psth_results_means,[5 95]);
%     fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'k', 'FaceAlpha',face_alpha, 'EdgeColor','none')
%     hold on
%     pctl2plot =  prctile(com_psth_results_means,[10 90]);
%     fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'k', 'FaceAlpha',face_alpha, 'EdgeColor','none')
%     pctl2plot =  prctile(com_psth_results_means,[1 99]);
%     fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'k', 'FaceAlpha',face_alpha, 'EdgeColor','none')
%     hold on
%     plot(time2use, median(com_psth_results_means, 'omitmissing'), 'Color', 'k')
    ylabel(labels(what2plot(j)))
end

%%


figure
face_alpha = 0.2;;
what2plot = find(indexes);
collected_means = [];
types2plot = [ 2 3];
for j=1:numel(what2plot)
    subplot(5,6,j)
    com_psth_results_means = squeeze(mean(com_psth_results(min_per_type_accepted,what2plot(j),types2plot(1),:),2, 'omitmissing'))-squeeze(mean(com_psth_results(min_per_type_accepted,what2plot(j),types2plot(2),:),2, 'omitmissing'));



 pctl2plot =  prctile(com_psth_results_means,[5 95]);
    fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'r', 'FaceAlpha',face_alpha, 'EdgeColor','none')
    hold on
    pctl2plot =  prctile(com_psth_results_means,[10 90]);
    fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'r', 'FaceAlpha',face_alpha, 'EdgeColor','none')
    % pctl2plot =  prctile(com_psth_results_means,[1 99]);
    % fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'r', 'FaceAlpha',face_alpha, 'EdgeColor','none')
    hold on
    plot(time2use, median(com_psth_results_means, 'omitmissing'), 'Color', 'r')
    collected_means = [collected_means;zscore(median(com_psth_results_means, 'omitmissing'))]
% 
%  pctl2plot =  prctile(com_psth_results_means,[5 95]);
%     fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'k', 'FaceAlpha',face_alpha, 'EdgeColor','none')
%     hold on
%     pctl2plot =  prctile(com_psth_results_means,[10 90]);
%     fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'k', 'FaceAlpha',face_alpha, 'EdgeColor','none')
%     pctl2plot =  prctile(com_psth_results_means,[1 99]);
%     fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'k', 'FaceAlpha',face_alpha, 'EdgeColor','none')
%     hold on
%     plot(time2use, median(com_psth_results_means, 'omitmissing'), 'Color', 'k')
    ylabel(labels(what2plot(j)))
    ylim([-10 10])
end

%%
figure
subplot(5,1,1:3)
imagesc(collected_means)


subplot(5,1,4:5)
[~, ~, ci] = ttest(collected_means);
fill([time2use fliplr(time2use)], [ci(1,:) fliplr(ci(2,:) )], 'k', 'FaceAlpha', .2, 'EdgeColor', 'None')
hold on

plot(time2use, mean(collected_means), 'k')
%%


selected_behaviors1              =  [1 4 5 6 7 8 9 14 15 18 19 20 21 24 25];% all play
% selected_behaviors1              =  [ 4 14  18 20 24];% locomotive self
% selected_behaviors1              =  [ 4 14  18 20 24]+1;% locomotive other
% selected_behaviors1              =  [6 8];% nonlocomotive
% selected_behaviors1 = 1;
% selected_behaviors1              =  [ 4 6 8 10 12 14 16 18 20 22 24 26];% self

disp(['Group 1', labels(selected_behaviors1)])
% selected_behaviors2              =  [10 12 16 22 26];% non play self
selected_behaviors2              =  [10 11 12 13 16 17 22 23 26 27];% all non play 
% selected_behaviors2              =  [6 8];% non locomotive play self
% selected_behaviors2              =  [6 8]+1;% non locomotive play other
% selected_behaviors2              =  [ 4 6 8 10 12 14 16 18 20 22 24 26]+1;% other
% selected_behaviors2                 = [10:13]
disp(['Group 2', labels(selected_behaviors2)])

indexes_g1 = false(numel(psth_list),1);
indexes_g1(selected_behaviors1)=true;  
pctlstplot = [10 90];

indexes_g2 = false(numel(psth_list),1);
indexes_g2(selected_behaviors2)=true;  


figure

for j=1:3
    subplot(1,3,j)
    com_psth_results_means = squeeze(median(com_psth_results(:,indexes_g1,j,:),2, 'omitmissing'));
    % com_psth_results_means = exp(squeeze(median(com_psth_results(:,indexes_g1,j,:),2, 'omitmissing')))/(1+exp(squeeze(median(com_psth_results(:,indexes_g1,j,:),2, 'omitmissing'))));




 pctl2plot =  prctile(com_psth_results_means,[5 95]);
    fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'r', 'FaceAlpha',face_alpha, 'EdgeColor','none')
    hold on
    pctl2plot =  prctile(com_psth_results_means,[10 90]);
    fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'r', 'FaceAlpha',face_alpha, 'EdgeColor','none')
    % pctl2plot =  prctile(com_psth_results_means,[1 99]);
    % fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'r', 'FaceAlpha',face_alpha, 'EdgeColor','none')
    hold on
    plot(time2use, median(com_psth_results_means, 'omitmissing'), 'Color', 'r')
% 

com_psth_results_means = squeeze(median(com_psth_results(:,indexes_g2,j,:),2, 'omitmissing'));
pctl2plot =  prctile(com_psth_results_means,[5 95]);
    fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'b', 'FaceAlpha',face_alpha, 'EdgeColor','none')
    hold on
    pctl2plot =  prctile(com_psth_results_means,[10 90]);
    fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'b', 'FaceAlpha',face_alpha, 'EdgeColor','none')
    % pctl2plot =  prctile(com_psth_results_means,[1 99]);
    % fill([time2use fliplr(time2use)], [pctl2plot(1,:) fliplr(pctl2plot(2,:))],'b', 'FaceAlpha',face_alpha, 'EdgeColor','none')
    hold on
    plot(time2use, median(com_psth_results_means, 'omitmissing'), 'Color', 'b')
end
