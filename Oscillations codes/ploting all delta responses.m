saving_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\Theta psth';

load([saving_folder,'\all_animals_delta_onset.mat'],'all_animals_delta_onset', 'all_animal_names', 'psth_time');


%%
 animal_name = cell(size(all_animal_names,1),1);

 for j=1:size(all_animal_names,1)
     this_session =all_animal_names{j,1};
     this_session = strsplit(this_session, ' ');
     this_animal = this_session{1};
     animal_name{j} = this_animal;
 end



%%

electorde_index = ismember([all_animal_names{:,2}],1);

[sorted_animals, animal_order] = sort(animal_name);
electorde_index = electorde_index(animal_order);
figure
matrix2plot = all_animals_delta_onset(animal_order,:);
matrix2plot = matrix2plot(electorde_index,:);
selected_animals = sorted_animals(electorde_index);
subplot(5,1,1:2)
imagesc(psth_time, 1:numel(selected_animals),matrix2plot)
axis xy
yticks(1:numel(selected_animals))
    yticklabels(selected_animals)
clim([-1 1])
xlim([-4 6])


subplot(5,1,3:5)
[~, ~, ci] = ttest(matrix2plot );
plot(psth_time,matrix2plot , ':k')
hold on
fill([psth_time fliplr(psth_time)],[ci(1,:) fliplr(ci(2,:))], 'k', 'FaceAlpha',.2, 'EdgeColor','none')
plot(psth_time,mean(matrix2plot ), 'k')

hold on
plot([psth_time(1) psth_time(end)],[0 0], 'k')

axis xy
ylim([-1 2])
xlim([-3 6])

%%

time2analyze = psth_time>=-3 & psth_time<=6;
sub_psth_time = psth_time(time2analyze);

results = testRepeatedOverTime(categorical(selected_animals), matrix2plot(:,time2analyze), 0);

%%
figure

subplot(5,1,1:2)
imagesc(psth_time, 1:numel(selected_animals),matrix2plot)
axis xy
yticks(1:numel(selected_animals))
    yticklabels(selected_animals)
    colorbar('northoutside')
clim([-1 1])
xlim([-3 6])

set(gca,'TickDir','out');


subplot(5,1,3:5)
plot(psth_time,matrix2plot , ':k')
hold on
plot(sub_psth_time, results.Intercept, 'k')
hold on
y_lim = ylim;
h = (results.pValue<0.05)';



start_end = sub_psth_time([find(diff([0,h,0])==1)' find(diff([0,h,0])==-1)'-1]) ;

for j=1:size(start_end,1)

    fill([start_end(j,[1 2 2 1])],y_lim([1 1 2 2]), 'r', 'FaceAlpha',.25, 'EdgeColor','none')
end

xlim([-3 6])
ylim([-1 2])
set(gca,'TickDir','out');
set(gca, 'FontSize', 24)