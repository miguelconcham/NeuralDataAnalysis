
two_animals_data = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\TWO ANIMALS';
saving_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\Theta psth';
animal_list = dir(two_animals_data);
animal_list(1:2) = [];

n_strctut = 1;
mi_structure = [];
animal_names = [];
%%
for fn = 1:numel(animal_list)
    
    if fn==1
        mi_structure = GENERATE_STRUCUTRE_animal_synch([two_animals_data, '\', animal_list(fn).name])
         
        n_strctut = n_strctut+numel(mi_structure);
        animal_names = [animal_names;[repmat(animal_list(fn).name,numel(mi_structure),1) num2cell(1:numel(mi_structure))']]
    else
        transt_psth = GENERATE_STRUCUTRE_animal_synch([two_animals_data, '\', animal_list(fn).name] );
      
        for sub_j=1:numel(transt_psth)
    
            mi_structure(n_strctut) = transt_psth(sub_j);
            n_strctut = n_strctut+1;
        end
        animal_names = [animal_names;[repmat(animal_list(fn).name,numel(transt_psth),1) num2cell(1:numel(transt_psth))']]


    end


end

%%

disp('saving')
save([saving_folder,'\mi_structure_20bins_delta.mat'],'mi_structure');
save([saving_folder,'\animal_names_mi_structure_20bins_delta.mat'],'animal_names');

%%   mergind data

all_psth_onset          = [];
all_pb_length           = [];
mean_delta_this_animal  = [];
all_mi_t                = [];
all_mi_bc               = [];
all_mi_pct              = [];
all_mi_pct2             = [];
all_mi_zscored          = [];
global_mi_distr_zscored = [];
baseline_range          = [-2 0];
global_mi               = [];
global_mi_distr         = [];
global_mi_pctl          = [];

for ns=1:numel(mi_structure)

    psth_zscored = mi_structure(ns).psth_zscored;


    global_mi               = [global_mi;mi_structure(ns).mi_global];
    global_mi_distr         = [global_mi_distr;mi_structure(ns).MI_rand_global'];

            all_mi_zscored          = [all_mi_zscored;(mi_structure(ns).mi_global-mean(mi_structure(ns).MI_rand_global))/std(mi_structure(ns).MI_rand_global)];
    global_mi_distr_zscored = [global_mi_distr_zscored;zscore(mi_structure(ns).MI_rand_global)'];

    this_pctl               = sum(mi_structure(ns).MI_rand_global>mi_structure(ns).mi_global)/numel(mi_structure(ns).MI_rand_global);
    global_mi_pctl          = [global_mi_pctl;this_pctl];
    
    PlayBout_table = mi_structure(ns).play_bouts_table;
    pb_length = diff(PlayBout_table')';
    psth_time = mi_structure(ns).psth_time;
    for j=1:numel(pb_length)
        psth_zscored(1,j, psth_time>pb_length(j)) = NaN;
        psth_zscored(2,j, psth_time>pb_length(j)) = NaN;
    end
    mean_delta_this_animal = [mean_delta_this_animal;[mean(squeeze(psth_zscored(1,:,:)),'omitmissing');mean(squeeze(psth_zscored(2,:,:)),'omitmissing')]];
    psth_zscored = [squeeze(psth_zscored(1,:,:));squeeze(psth_zscored(2,:,:))];
    all_psth_onset = [all_psth_onset;psth_zscored];

    all_pb_length = [all_pb_length;[pb_length;pb_length]];

    mi_time = mi_structure(ns).mi_time;
    this_mi_t = mi_structure(ns).mi_t';
    all_mi_t = [all_mi_t;mi_structure(ns).mi_t'];
    baseline_range_index = mi_time>=baseline_range(1) & mi_time<=baseline_range(2);
    this_mi_t_bc = (this_mi_t - mean(this_mi_t(baseline_range_index)))/std(this_mi_t(baseline_range_index));
    all_mi_bc = [all_mi_bc;this_mi_t_bc];
    all_mi_pct = [all_mi_pct;mi_structure(ns).mi_t_pctl'];
    mi_t_pctl_2 =this_mi_t;
    MI_t_rand_play = mi_structure(ns).MI_t_rand_play;
    for t=1:numel(this_mi_t)
    mi_t_pctl_2(t) = sum(MI_t_rand_play>this_mi_t(t))/numel(MI_t_rand_play);
    end

     all_mi_pct2 = [all_mi_pct2;mi_t_pctl_2];

end

%% plot a psth
ns =4;

this_play_bout = mi_structure(ns).play_bouts_table;
these_lengths = diff(this_play_bout')';
[sorted_lengths, order] = sort(these_lengths);

psth_zscored    = mi_structure(ns).psth_zscored;
psth_time       = mi_structure(ns).psth_time;
x_lim = mi_structure(ns).global_time_range;

figure
subplot(1,2,1)
imagesc(psth_time, 1:size(psth_zscored,2), squeeze(psth_zscored(1,order,:)))
axis xy
xlim(x_lim)
hold on
clim([-2 2])
plot(sorted_lengths, 1:size(psth_zscored,2), 'w')
plot(sorted_lengths*0, 1:size(psth_zscored,2), 'w')

subplot(1,2,2)
imagesc(psth_time, 1:size(psth_zscored,2), squeeze(psth_zscored(2,order,:)))
axis xy
xlim(x_lim)
hold on
plot(sorted_lengths, 1:size(psth_zscored,2), 'w')
plot(sorted_lengths*0, 1:size(psth_zscored,2), 'w')
clim([-2 2])

%% plot global mi distribution and observations


figure
hist_ranges = round([min(global_mi_distr_zscored(:)) max(global_mi_distr_zscored(:))],2)
hold on
for j=1:size(global_mi_distr_zscored,1)
histogram(global_mi_distr_zscored(j,:),hist_ranges(1):0.05:hist_ranges(2),'FaceColor', 'k', 'FaceAlpha',.25, 'EdgeColor', 'none')
end

% plot([all_mi_zscored';all_mi_zscored'], [0 50], 'r')
% plot(all_mi_zscored, all_mi_zscored*0, '.r', 'MarkerSize',12)
% text( all_mi_zscored, all_mi_zscored*0 + 60, strsplit(num2str(round(global_mi_pctl,3)'), ' '),'Color',[1 0 0])

%%
figure

subplot(5,1,3:5)
plot(psth_time,mean_delta_this_animal, 'k','LineWidth',.05)

hold on
[~,~, ci] = ttest(mean_delta_this_animal);
fill([psth_time fliplr(psth_time)],[ci(1,:), fliplr(ci(2,:))],'r', 'FaceAlpha',.2, 'EdgeColor','none')
plot(psth_time,mean(mean_delta_this_animal, 'omitmissing'), 'r', 'LineWidth',2)
plot([psth_time(1) psth_time(end)], [0 0], 'k')
xlim([-3 4])
ylim([-1 2])
%%

figure
imagesc(mi_time,1:size(all_mi_bc,1),all_mi_bc)
axis xy
clim([- 1 2])
xlim([-3 4])
yticks(1:size(animal_names,1))
yticklabels(animal_names(:,1))


%%

figure
% plot(mi_time,all_mi_bc, ':k')
subplot(1,2,1)
hold on
[~,~, ci] = ttest(all_mi_bc);
fill([mi_time fliplr(mi_time)],[ci(1,:), fliplr(ci(2,:))],'k', 'FaceAlpha',.2, 'EdgeColor','none')
plot(mi_time,mean(all_mi_bc, 'omitmissing'), 'k', 'LineWidth',2)
ylim([-2 4])
yyaxis right
[~,~, ci] = ttest(mean_delta_this_animal);
fill([psth_time fliplr(psth_time)],[ci(1,:), fliplr(ci(2,:))],'r', 'FaceAlpha',.2, 'EdgeColor','none')
plot(psth_time,mean(mean_delta_this_animal, 'omitmissing'), 'r', 'LineWidth',2)
hold on
plot([psth_time(1) psth_time(end)], [0 0], 'k')
xlim([-3 4])
ylim([-1 2])




subplot(1,2,2)
% plot(mi_time,1-all_mi_pct, ':k')
hold on
plot(mi_time,mean(1-all_mi_pct, 'omitmissing'), 'k')
ylim([.5 1])
plot([psth_time(1) psth_time(end)], [0.95 0.95], 'r')
yyaxis right
[~,~, ci] = ttest(mean_delta_this_animal);
fill([psth_time fliplr(psth_time)],[ci(1,:), fliplr(ci(2,:))],'r', 'FaceAlpha',.2, 'EdgeColor','none')
plot(psth_time,mean(mean_delta_this_animal, 'omitmissing'), 'r', 'LineWidth',2)
plot([psth_time(1) psth_time(end)], [0 0], 'k')

hold on
ylim([-1 2])

xlim([-3 4])
%%

figure
plot(mi_time,1-all_mi_pct2, ':k')
hold on
plot(mi_time,mean(1-all_mi_pct2, 'omitmissing'), 'k')
ylim([.5 1])
yyaxis right
[~,~, ci] = ttest(mean_delta_this_animal);
fill([psth_time fliplr(psth_time)],[ci(1,:), fliplr(ci(2,:))],'r', 'FaceAlpha',.2, 'EdgeColor','none')
plot(psth_time,mean(mean_delta_this_animal, 'omitmissing'), 'r', 'LineWidth',2)
plot([psth_time(1) psth_time(end)], [0 0], 'k')
hold on
ylim([-1 2])



%%

figure
plot(mi_time,movmean(100*(sum(all_mi_pct2<0.1)/size(all_mi_pct2,1)),20), 'k')
% hold on
% plot(mi_time,mean(1-all_mi_pct, 'omitmissing'), 'k')
% ylim([.5 1])
% yyaxis right
% [~,~, ci] = ttest(mean_delta_this_animal);
% fill([psth_time fliplr(psth_time)],[ci(1,:), fliplr(ci(2,:))],'r', 'FaceAlpha',.2, 'EdgeColor','none')
% plot(psth_time,mean(mean_delta_this_animal, 'omitmissing'), 'r', 'LineWidth',2)
% plot([psth_time(1) psth_time(end)], [0 0], 'k')
% hold on
% ylim([-1 2])


%% all delta animals

sub_index = ismember(time,psth_time);

all_animals_delta_onset_overlap = all_animals_delta_onset(:,sub_index);

sub_index = ismember(psth_time,time);


mean_delta_this_animal_overlap = mean_delta_this_animal(:,sub_index);

all_animals_delta_onset = [all_animals_delta_onset_overlap;mean_delta_this_animal_overlap] ;



animal_names_psth_code = animal_names;

animal_names_sync_code = {'PD3C1 0620 C1','PD3C2 0620 C2','PD3C1 0623 C1','PDPD1 0623 PD1','B1S3 1019 Single','B1D1 1019 Dual','B1S3 1020 Single','B1D1 1020 Dual','B2S2 1105 Single2','B2S2 1105 Single4','B2S2 1107 Single2','B2S2 1107 Single4'}';

animal_names_sync_code = [animal_names_sync_code, repmat({1}, numel(animal_names_sync_code),1)];

all_animal_names = [animal_names_psth_code;animal_names_sync_code];

save([saving_folder,'\all_animals_delta_onset.mat'],'all_animals_delta_onset', 'all_animal_names', 'psth_time');
%%


%% animal selection


