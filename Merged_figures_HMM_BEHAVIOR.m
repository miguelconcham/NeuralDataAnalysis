%% mergin transition matrixes (working on)
transition_matrix_files = dir('*3states_tm*');
all_transitions = nan(numel(transition_matrix_files), 3, 3);
transition_bar  = nan(numel(transition_matrix_files), 9);
combinations = nan(2,9);

n_bar = 1;
for j=1:3
    for i=1:3
        combinations(:,n_bar) = [i ;j];
               n_bar = n_bar+1;
    end
end

for fn=1:numel(transition_matrix_files)
    a = readNPY(transition_matrix_files(fn).name)
    a = a(re_assignment(fn,:),re_assignment(fn,:));
    all_transitions(fn,:,:)= a;
    n_bar = 1;
    for j=1:3
        for i=1:3
            transition_bar(fn,n_bar) = a(i,j);
            n_bar = n_bar+1;
        end
    end

end


figure
subplot(1,2,1)
plot([1 2 3],transition_bar(:, [1 5 9]).^100, 'k.')
hold on
plot(repmat([1 2 3],7,1)',(transition_bar(:, [1 5 9]))'.^100, 'k:')
hold on
plot([1 2 3],mean(transition_bar(:, [1 5 9]).^100), '_r')
xlim([.5 3.5])

subplot(1,2,2)

plot(1:6,transition_bar(:, [2  4 6  8 3 7]), 'k.')
hold on
plot(repmat(1:6,7,1)',(transition_bar(:, [2  4 6  8 3 7]))', ':k')
plot(1:6,mean(transition_bar(:, [2  4 6  8 3 7])), '_r')
xlim([.5 6.5])
xtics(1:6)
xticklabels({'NO-Play 2 Transition', 'NO-Play 2 Transition','Play 2 Transition','Play 2 Transition', 'Play and no play','Play and no play' })





%% 1 merging confusing matrix and ploting
hmm_raw_data    = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\HMM 2 and 3 states 2 partners';
cd(hmm_raw_data)
confusion_matrix_files = dir('*confusion_matrix*');

all_confusion_matrix = nan(numel(confusion_matrix_files),4);

for fn= 1:numel(confusion_matrix_files)
    load(confusion_matrix_files(fn).name)
all_confusion_matrix(fn,:) = confusion_matrix;
end

figure
plot([1 2 3 4], all_confusion_matrix, 'k.')
hold on
plot([1 2 3 4], all_confusion_matrix, 'k:')

%% 2 loading properites between play and non play 

prediction_struct_files = dir('* prediction_struct*');
is_there_play_bout      = [];
is_there_hmm            = [];
is_this_hmm             = [];
filled_play_bouts       = [];
tripple_states          = [];
re_assignment           = [];
is_there_play_beh       = [];



total_number_of_hmm = 0;

for fn= 1:numel(prediction_struct_files)
     load(prediction_struct_files(fn).name) 
     is_there_play_bout      = [is_there_play_bout;prediction_struct.is_there_play_bout];
     is_there_play_beh       = [is_there_play_beh;prediction_struct.is_there_play_beh];
     is_there_hmm            = [is_there_hmm;prediction_struct.is_there_hmm];
     is_this_hmm             = [is_this_hmm;prediction_struct.is_this_hmm];
     filled_play_bouts       = [filled_play_bouts;prediction_struct.filled_play_bouts];
     tripple_states          = [tripple_states;prediction_struct.what_3states_is];
     re_assignment           = [re_assignment;prediction_struct.re_assignment]
end
% current_hmm = tripple_states==3;
psth_edges = prediction_struct.psth_edges;
%% plot hmm, pb, and  hmm no-pb


figure('units','normalized','outerposition',[0 0 .5 1]);
% colormap(1-gray)
% current_hmm =  tripple_states==3;
current_hmm = is_this_hmm;
filled_hmm_states = filled_play_bouts;
[hmm_length_ordered, pb_order] = sort(diff(filled_hmm_states'));
shorter = diff(filled_hmm_states')<1.5
longer  = diff(filled_hmm_states')>1.5
properly_labeled = filled_hmm_states(pb_order,1)<=Inf;

%hmm
subplot(5,4,[1 5 9])
imagesc(psth_edges, 1:sum(properly_labeled), 1- cat(3,current_hmm(pb_order(properly_labeled),:), is_there_play_bout(pb_order(properly_labeled),:)*0 , current_hmm(pb_order(properly_labeled),:)*0))
axis xy
hold on
plot([0 0], [.5 sum(properly_labeled)+.5], 'k')
plot(hmm_length_ordered,1:sum(properly_labeled),'k')
yticks([])
xticks(psth_edges(1):psth_edges(end))
title('Hidden states')

subplot(5,4,[13 17])
plot(psth_edges, mean(current_hmm), 'k')
hold on
plot(psth_edges, mean(current_hmm(shorter,:))', 'b')
plot(psth_edges, mean(current_hmm(longer,:)), 'r')


%playbout
subplot(5,4,[1 5 9]+1)
imagesc(psth_edges, 1:sum(properly_labeled), 1-cat(3,current_hmm(pb_order(properly_labeled),:)*0, is_there_play_bout(pb_order(properly_labeled),:), is_there_play_beh(pb_order(properly_labeled),:)*0))
axis xy
hold on
plot([0 0], [.5 sum(properly_labeled)+.5], 'k')
plot(hmm_length_ordered,1:sum(properly_labeled),'k')
yticks([])
xticks(psth_edges(1):psth_edges(end))
title('Play bout')

subplot(5,4,[13 17]+1)
plot(psth_edges, mean(is_there_play_bout), 'k')
hold on
plot(psth_edges, mean(is_there_play_bout(shorter,:))', 'b')
plot(psth_edges, mean(is_there_play_bout(longer,:)), 'r')

%%play bahavior
subplot(5,4,[1 5 9]+2)
imagesc(psth_edges, 1:sum(properly_labeled), 1-cat(3,current_hmm(pb_order(properly_labeled),:)*0, is_there_play_bout(pb_order(properly_labeled),:)*0, is_there_play_beh(pb_order(properly_labeled),:)))
axis xy
hold on
plot([0 0], [.5 sum(properly_labeled)+.5], 'k')
plot(hmm_length_ordered,1:sum(properly_labeled),'k')
yticks([])
xticks(psth_edges(1):psth_edges(end))
title('Indivudual Play behaviors')

subplot(5,4,[13 17]+2)
plot(psth_edges, mean(is_there_play_beh), 'k')
hold on
plot(psth_edges, mean(is_there_play_beh(shorter,:))', 'b')
plot(psth_edges, mean(is_there_play_beh(longer,:)), 'r')

%playbout & hmm
subplot(5,4,[1 5 9]+3)
imagesc(psth_edges, 1:sum(properly_labeled),1- cat(3,current_hmm(pb_order(properly_labeled),:), is_there_play_bout(pb_order(properly_labeled),:) , is_there_play_beh(pb_order(properly_labeled),:)))
axis xy
hold on
plot([0 0], [.5 sum(properly_labeled)+.5], 'k')
plot(hmm_length_ordered,1:sum(properly_labeled),'k')
title('Overlap')
subplot(5,4,[13 17]+3)
hold on
plot(psth_edges, mean(is_there_play_bout(pb_order(properly_labeled),:) & current_hmm(pb_order(properly_labeled),:)), 'k')
plot(psth_edges, mean(is_there_play_bout(shorter,:) & current_hmm(shorter,:)), 'b')
plot(psth_edges, mean(is_there_play_bout(longer,:) & current_hmm(longer,:)), 'r')


%% ploting 3states align to 2state onset

figure('units','normalized','outerposition',[0 0 .5 1]);
colormap(gray)
imagesc(psth_edges, 1:sum(properly_labeled), tripple_states(pb_order(properly_labeled),:))
axis xy
hold on
plot([0 0], [.5 sum(properly_labeled)+.5], 'k')
plot(hmm_length_ordered,1:sum(properly_labeled),'k')
yticks([])

%% 4 mergin behavior onsets and offsets


behavior_onset_offset_struct_files = dir('*behavior_onset_offset_struct*');


behavior_type_list = [];
total_number_of_hmm = 0;
total_number_of_hmm_3states = 0;

for fn= 1:numel(behavior_onset_offset_struct_files)
     load(behavior_onset_offset_struct_files(fn).name) 
     behavior_type_list = [behavior_type_list; behavior_onset_offset_struct.behavior_tpes];
     total_number_of_hmm = total_number_of_hmm+size(behavior_onset_offset_struct.filled_play_bouts,1);
    total_number_of_hmm_3states = total_number_of_hmm_3states++size(behavior_onset_offset_struct.filled_hmm_3states,1);

end
behavior_type_list = unique(behavior_type_list);
behavior_type_list(ismember(behavior_type_list, {'Partners session', 'SA'})) = [];


merged_behaviors_onset  = zeros(numel(behavior_type_list),total_number_of_hmm,size(behavior_onset_offset_struct.behavior_offset,3));
merged_behaviors_offset = zeros(numel(behavior_type_list),total_number_of_hmm,size(behavior_onset_offset_struct.behavior_offset,3));


merged_behaviors_onset_3states  = zeros(3,numel(behavior_type_list),total_number_of_hmm_3states,size(behavior_onset_offset_struct.behavior_offset,3));
merged_behaviors_offset_3states = zeros(3,numel(behavior_type_list),total_number_of_hmm_3states,size(behavior_onset_offset_struct.behavior_offset,3));
all_hmm_3states = [];
all_hmm_lengths = [];
total_number_of_hmm = 0;
total_number_of_hmm_3states = 0;
for fn= 1:numel(behavior_onset_offset_struct_files)
    load(behavior_onset_offset_struct_files(fn).name)
    n_hmm = size(behavior_onset_offset_struct.filled_play_bouts,1);
    n_hmm_3states = size(behavior_onset_offset_struct.filled_hmm_3states,1);
    all_hmm_lengths = [all_hmm_lengths; diff(behavior_onset_offset_struct.filled_play_bouts')'];

    this_3_States_matrix = behavior_onset_offset_struct.filled_hmm_3states;

    for j=1:3
        this_3_States_matrix (behavior_onset_offset_struct.filled_hmm_3states(:,1)==j-1,1) = re_assignment(fn,j)-1;
    end
    all_hmm_3states = [all_hmm_3states;this_3_States_matrix];

    current_behaviors = find(ismember(behavior_type_list,behavior_onset_offset_struct.behavior_tpes));

    for beavior_present = current_behaviors'
        beh_index = ismember(behavior_onset_offset_struct.behavior_tpes,behavior_type_list(beavior_present));
        merged_behaviors_onset(beavior_present,total_number_of_hmm+1:total_number_of_hmm+n_hmm,:) = behavior_onset_offset_struct.behavior_onset(beh_index,:,:);
        merged_behaviors_offset(beavior_present,total_number_of_hmm+1:total_number_of_hmm+n_hmm,:) = behavior_onset_offset_struct.behavior_offset(beh_index,:,:);

        merged_behaviors_onset_3states(:,beavior_present,total_number_of_hmm_3states+1:total_number_of_hmm_3states+n_hmm_3states,:) = behavior_onset_offset_struct.behavior_onset_3states(re_assignment(fn,:), beh_index,:,:);
        merged_behaviors_offset_3states(:,beavior_present,total_number_of_hmm_3states+1:total_number_of_hmm_3states+n_hmm_3states,:) = behavior_onset_offset_struct.behavior_offset_3states(re_assignment(fn,:),beh_index,:,:);


    end
    total_number_of_hmm=total_number_of_hmm+n_hmm;
    total_number_of_hmm_3states = total_number_of_hmm_3states+n_hmm_3states;
    
end
%% 5 behavioral_composition for 3 states

proportion_of_behavior = nan(numel(behavior_type_list),3);
for bn = 1:numel(behavior_type_list)

behavior_bool = squeeze(merged_behaviors_onset( bn,:,:)>0);

proportion_of_behavior(bn,:) = [sum(sum(tripple_states==3 & behavior_bool)) sum(sum(tripple_states==2 & behavior_bool))  sum(sum(tripple_states==1 & behavior_bool)) ]/sum(sum(behavior_bool));
end

figure
subplot(1,2,1)
[~, order] = sort(proportion_of_behavior(:,1));

var2include = [1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 ]
ordered_proportion =proportion_of_behavior(order,:);
ordered_labels = behavior_type_list(order);
bar(ordered_proportion(var2include,:), 'stacked')
xticks( 1:numel(ordered_labels(var2include)))
xticklabels(ordered_labels(var2include))

subplot(1,2,2)
[~, order] = sort(proportion_of_behavior(:,1)+proportion_of_behavior(:,2));
var2include = [1 2 3 4 5 6 7  9 10 11 12 13 14 15 16 17 ]
ordered_proportion =proportion_of_behavior(order,:);
ordered_labels = behavior_type_list(order);
bar(ordered_proportion(var2include,:), 'stacked')
xticks( 1:numel(ordered_labels(var2include)))
xticklabels(ordered_labels(var2include))
%% 6 old bar plot of 2 states loading play behavior struct and ploting (bar plot)

play_behavior_struct_files = dir('*play_behavior_struct*');




all_behavior_types = [];

for fn= 1:numel(play_behavior_struct_files)
     load(play_behavior_struct_files(fn).name) 
     all_behavior_types = [all_behavior_types; play_behavior_struct.behavior_types];

end
all_behavior_types = unique(all_behavior_types);

all_numbers         = zeros(numel(all_behavior_types),1);
all_behavior_count  = zeros(numel(all_behavior_types),3);

for fn= 1:numel(play_behavior_struct_files)
    load(play_behavior_struct_files(fn).name) 
    behavior_count = diag(play_behavior_struct.numbers)*play_behavior_struct.proportions;

    for bn = find(ismember(all_behavior_types,play_behavior_struct.behavior_types))'
        beh_index = ismember(play_behavior_struct.behavior_types, all_behavior_types(bn));
        all_numbers(bn) = all_numbers(bn)+play_behavior_struct.numbers(beh_index);

        all_behavior_count(bn,:) =  all_behavior_count(bn,:)+behavior_count(beh_index,:);
    end
end

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
xticklabels(all_behavior_types_realNames(bar_order(toincludemove)))
axis tight

%% now ploting (2states)
[hmm_length_ordered, hmm_order] = sort(all_hmm_lengths);
psth_edges = behavior_onset_offset_struct.psth_edges;
behavior_type_list_real_names = {'not labeled','Bite','Boxing','PlayfulApproach','Chasing','nonPlayfulApproach','Escape'...
    ,'Evasion','Grooming','Pin','PounceNeck','PounceNeckInmov','PounceBack','PounceBackInmov','Rearing','Scratch','Sniffing', 'SniffingCable'};

for bn =1:numel(behavior_type_list)

     figure('units','normalized','outerposition',[0 0 .5 1]);
    colormap(1-gray)
    subplot(5,2,[1 3 5])

    imagesc(psth_edges,1:numel(hmm_length_ordered), squeeze(merged_behaviors_onset(bn,hmm_order,:)))
    hold on
    plot([0 0],[1 numel(hmm_length_ordered)],'r')
    hold on
    plot(hmm_length_ordered,1:numel(hmm_length_ordered),'r')
    axis xy
    title(behavior_type_list_real_names{bn})
    subplot(5,2,[7 9])

    plot(psth_edges,mean(100* squeeze(merged_behaviors_onset(bn,:,:))), 'k')
    hold on
    plot([0 0],[0 max(squeeze(100* mean(merged_behaviors_onset(bn,:,:))))],'r')
    axis tight


    subplot(5,2,[1 3 5]+1)

    imagesc(psth_edges,1:numel(hmm_length_ordered), squeeze(merged_behaviors_offset(bn,hmm_order,:)))
    hold on
    plot([0 0],[1 numel(hmm_length_ordered)],'r')
    hold on
    plot(-hmm_length_ordered,1:numel(hmm_length_ordered),'r')
    axis xy
    title(behavior_type_list_real_names{bn})
    subplot(5,2,[7 9]+1)

    plot(psth_edges,squeeze(100* mean(merged_behaviors_offset(bn,:,:))), 'k')
    hold on
    plot([0 0],[0 max(squeeze(100* mean(merged_behaviors_offset(bn,:,:))))],'r')
    axis tight

end

%% now ploting two states play and non playful hmm states

playful_hmm = any(is_this_hmm & is_there_play_beh,2);
[hmm_length_ordered, hmm_order] = sort(all_hmm_lengths);
playful_hmm = playful_hmm(hmm_order);

psth_edges = behavior_onset_offset_struct.psth_edges;
behavior_type_list_real_names = {'not labeled','Bite','Boxing','PlayfulApproach','Chasing','nonPlayfulApproach','Escape'...
    ,'Evasion','Grooming','Pin','PounceNeck','PounceNeckInmov','PounceBack','PounceBackInmov','Rearing','Scratch','Sniffing', 'SniffingCable'};

for bn =1:numel(behavior_type_list)

     figure('units','normalized','outerposition',[0 0 .5 1]);
    colormap(1-gray)
    subplot(4,2,1 )
    playful_matrix =  squeeze(merged_behaviors_onset(bn,hmm_order(playful_hmm),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(playful_hmm)),playful_matrix)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(playful_hmm))],'r')
    hold on
    plot(hmm_length_ordered(playful_hmm),1:numel(hmm_length_ordered(playful_hmm)),'r')
    axis xy
    title(behavior_type_list_real_names{bn})


     subplot(4,2,3 )
    non_playful_matrix =  squeeze(merged_behaviors_onset(bn,hmm_order(~playful_hmm),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(~playful_hmm)),non_playful_matrix)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(~playful_hmm))],'r')
    hold on
    plot(hmm_length_ordered(~playful_hmm),1:numel(hmm_length_ordered(~playful_hmm)),'r')
    axis xy
    title(behavior_type_list_real_names{bn})

    subplot(4,2,[5 7])
   plot(psth_edges,mean(100* non_playful_matrix), 'k')
   hold on
   plot(psth_edges,mean(100* playful_matrix), 'r')
   y_lim = ylim;
   y_lim(1) = 0;
    hold on
    plot([0 0],y_lim,'r')
    axis tight


    subplot(4,2,2)
     playful_matrix =  squeeze(merged_behaviors_offset(bn,hmm_order(playful_hmm),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(playful_hmm)), playful_matrix)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(playful_hmm))],'r')
    hold on
    plot(-hmm_length_ordered(playful_hmm),1:numel(hmm_length_ordered(playful_hmm)),'r')
    axis xy
    title(behavior_type_list_real_names{bn})
    subplot(4,2,[5 7]+1)


    subplot(4,2,4)
     non_playful_matrix =  squeeze(merged_behaviors_offset(bn,hmm_order(~playful_hmm),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(playful_hmm)), non_playful_matrix)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(~playful_hmm))],'r')
    hold on
    plot(-hmm_length_ordered(~playful_hmm),1:numel(hmm_length_ordered(~playful_hmm)),'r')
    axis xy
    title(behavior_type_list_real_names{bn})

    subplot(4,2,[5 7]+1)
   plot(psth_edges,mean(100* non_playful_matrix), 'k')
   hold on
   plot(psth_edges,mean(100* playful_matrix), 'r')
   y_lim = ylim;
   y_lim(1) = 0;
    hold on
    plot([0 0],y_lim,'r')
    axis tight
    pause(.1)

end


%% now ploting 3states



[hmm_length_ordered, pb_order] = sort(diff(all_hmm_3states(:,[2 3])'));
ordered_hmm_number = all_hmm_3states(pb_order,1);
x_lim = [-4 4];
for bn=1:numel(behavior_type_list)
    figure('units','normalized','outerposition',[0 0 1 1]);
    colormap(1-gray)
    for hmm_n = 1:3
        index = ordered_hmm_number==hmm_n-1;

        subplot(5,6,[1 7 13]+(hmm_n-1)*2)
        matrix2plot = squeeze(merged_behaviors_onset_3states(hmm_n,bn,pb_order,:));
        matrix2plot = matrix2plot(index,:);
        imagesc(psth_edges,1:numel(hmm_length_ordered(index)),matrix2plot )
        hold on
        plot([0 0],[1 numel(hmm_length_ordered(index))],'r')
        hold on
        plot(hmm_length_ordered(index),1:numel(hmm_length_ordered(index)),'r')
        axis xy
        xlim(x_lim)
        title(behavior_type_list{bn})

        subplot(5,6,[19 25]+(hmm_n-1)*2)
        plot(psth_edges,squeeze(100* mean(matrix2plot)), 'k')
        hold on
        plot([0 0],[0 max(squeeze(100* mean(matrix2plot)))],'r')
        axis tight
        xlim(x_lim)


        subplot(5,6,[1 7 13]+1+(hmm_n-1)*2)
        matrix2plot = squeeze(merged_behaviors_offset_3states(hmm_n,bn,pb_order,:));
        matrix2plot = matrix2plot(index,:);
        imagesc(psth_edges,1:numel(hmm_length_ordered(index)), matrix2plot)
        hold on
        plot([0 0],[1 numel(hmm_length_ordered(index))],'r')
        hold on
        plot(-hmm_length_ordered(index),1:numel(hmm_length_ordered(index)),'r')
        axis xy
        xlim(x_lim)
        title(behavior_type_list{bn})

        subplot(5,6,[19 25]+1+(hmm_n-1)*2)
        plot(psth_edges,squeeze(100* mean(matrix2plot)), 'k')
        hold on
        plot([0 0],[0 max(squeeze(100* mean(matrix2plot)))],'r')
        axis tight
        xlim(x_lim)
    end
    pause(.1)

end
%% load call struct


call_struct_files = dir('*call_struct*');

n_calls_per_hmm = [];
properties_list = [];
total_number_of_hmm = 0;

for fn= 1:numel(confusion_matrix_files)
     load(call_struct_files(fn).name) 
     properties_list = [properties_list; call_struct.properties2estimate'];
     total_number_of_hmm = total_number_of_hmm+size(call_struct.call_onset,1);

end
properties_list = unique(properties_list);

merged_call_onset               = zeros(total_number_of_hmm,size(call_struct.call_offset,2));
merged_call_offset              = zeros(total_number_of_hmm,size(call_struct.call_offset,2));
merged_properties_call_onset    = zeros(numel(properties_list),total_number_of_hmm,size(call_struct.properties_call_offset,3));
merged_properties_call_offset   = zeros(numel(properties_list),total_number_of_hmm,size(call_struct.properties_call_offset,3));


all_hmm_lengths = [];
total_number_of_hmm = 0;
for fn= 1:numel(confusion_matrix_files)
    load(call_struct_files(fn).name)
    n_hmm = size(call_struct.call_offset,1);
    all_hmm_lengths = [all_hmm_lengths; diff(call_struct.filled_play_bouts')'];
    current_properties = find(ismember(properties_list,call_struct.properties2estimate));
    merged_call_onset(total_number_of_hmm+1:total_number_of_hmm+n_hmm,:)  = call_struct.call_onset  ;
    merged_call_offset(total_number_of_hmm+1:total_number_of_hmm+n_hmm,:) = call_struct.call_offset  ;
    for property_present = current_properties'
        prop_index = ismember(call_struct.properties2estimate,properties_list(property_present));
        merged_properties_call_onset(property_present,total_number_of_hmm+1:total_number_of_hmm+n_hmm,:) = call_struct.properties_call_onset(prop_index,:,:);
        merged_properties_call_offset(property_present,total_number_of_hmm+1:total_number_of_hmm+n_hmm,:) = call_struct.properties_call_offset(prop_index,:,:);
    end
    total_number_of_hmm=total_number_of_hmm+n_hmm;
end
psth_edges = call_struct.psth_edges;
%% now plot call osnet offset


[hmm_length_ordered, pb_order] = sort(all_hmm_lengths);
mov_mean_window = 5;

figure
colormap(1-gray)
subplot(5,2,[1 3 5])

imagesc(psth_edges,1:numel(hmm_length_ordered), squeeze(merged_call_onset(pb_order,:)))
hold on
plot([0 0],[1 numel(hmm_length_ordered)],'r')
hold on
plot(hmm_length_ordered,1:numel(hmm_length_ordered),'r')
axis xy
title('HMM state onset')
subplot(5,2,[7 9])

plot(psth_edges,squeeze(100* mean(merged_call_onset)), 'k')
hold on
plot(psth_edges,movmean(squeeze(100* mean(merged_call_onset)),mov_mean_window), 'b')

plot([0 0],[0 max(squeeze(100* mean(merged_call_onset)))],'r')
axis tight
ylabel('% Call ')

colormap(1-gray)
subplot(5,2,[1 3 5]+1)

imagesc(psth_edges,1:numel(hmm_length_ordered), squeeze(merged_call_offset(pb_order,:)))
hold on
plot([0 0],[1 numel(hmm_length_ordered)],'r')
hold on
plot(-hmm_length_ordered,1:numel(hmm_length_ordered),'r')
axis xy
title('HMM state offset')
subplot(5,2,[7 9]+1)

plot(psth_edges,squeeze(100* mean(merged_call_offset)), 'k')
hold on
plot(psth_edges,movmean(squeeze(100* mean(merged_call_offset)),mov_mean_window), 'b')
hold on
plot([0 0],[0 max(squeeze(100* mean(merged_call_offset)))],'r')
ylabel('% Call ')
axis tight

%% now plot call property plot


for p_n = 1:numel(properties_list)

    figure('units','normalized','outerposition',[0 0 .5 1]);
    colormap(1-gray)
    subplot(5,2,[1 3 5])
    prop_matrix = squeeze(merged_properties_call_onset(p_n,pb_order,:));

    imagesc(psth_edges,1:numel(hmm_length_ordered), prop_matrix)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered)],'r')
    hold on
    plot(hmm_length_ordered,1:numel(hmm_length_ordered),'r')
    axis xy
       title(properties_list{p_n})
    subplot(5,2,[7 9])

    plot(psth_edges,squeeze(mean(prop_matrix, 'omitmissing')), 'k')
    hold on
    plot(psth_edges,movmean(squeeze(mean(prop_matrix, 'omitmissing')),mov_mean_window), 'b')

    plot([0 0],[min(squeeze(mean(prop_matrix, 'omitmissing'))) max(squeeze(mean(prop_matrix, 'omitmissing')))],'r')
    axis tight
    ylabel('% Call ')

    colormap(1-gray)
    subplot(5,2,[1 3 5]+1)
    prop_matrix = squeeze(merged_properties_call_offset(p_n,pb_order,:));
    imagesc(psth_edges,1:numel(hmm_length_ordered), prop_matrix)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered)],'r')
    hold on
    plot(-hmm_length_ordered,1:numel(hmm_length_ordered),'r')
    axis xy
    title(properties_list{p_n})
    subplot(5,2,[7 9]+1)

    plot(psth_edges,squeeze( mean(prop_matrix, 'omitmissing')), 'k')
    hold on
    plot(psth_edges,movmean(squeeze(mean(prop_matrix, 'omitmissing')),mov_mean_window), 'b')
    hold on
    plot([0 0],[min(squeeze(mean(prop_matrix, 'omitmissing'))) max(squeeze(mean(prop_matrix, 'omitmissing')))],'r')
    ylabel('% Call ')
    axis tight

    pause(.1)

end


%% loading variable plot

 variable_onset_struct_files = dir('* variable_onset_struct*');


variable_names = [];
total_number_of_hmm         = 0;
total_number_of_hmm_3states = 0;

for fn= 1:numel(variable_onset_struct_files)
     load(variable_onset_struct_files(fn).name) 
     
    
     if fn==1
        

     variable_names = variable_onset_struct.variable_types';
     end
     total_number_of_hmm = total_number_of_hmm+size(variable_onset_struct.beh_properties_onset,2);
     total_number_of_hmm_3states = total_number_of_hmm_3states+size(variable_onset_struct.beh_properties_onset_3states  ,3);
end



all_variable_onsets= nan(numel(variable_names),total_number_of_hmm,size(variable_onset_struct.beh_properties_onset,3));
all_variable_offsets= nan(numel(variable_names),total_number_of_hmm,size(variable_onset_struct.beh_properties_onset,3));



% all_variable_onsets_3states = nan(3,numel(variable_names),total_number_of_hmm,size(variable_onset_struct.beh_properties_onset,3));
% all_variable_offsets_3states= nan(3,numel(variable_names),total_number_of_hmm,size(variable_onset_struct.beh_properties_onset,3));



all_hmm_lengths = [];
% all_hmm_lengths_3states = [];
total_number_of_hmm         = 0;
% total_number_of_hmm_3states = 0;
for fn= 1:numel(variable_onset_struct_files)
     load(variable_onset_struct_files(fn).name) 

        n_hmm = size(variable_onset_struct.beh_properties_onset,2);
        % n_hmm_3stsates = size(variable_onset_struct.beh_properties_onset_3states,3);
        all_hmm_lengths = [all_hmm_lengths; diff(variable_onset_struct.filled_play_bouts')'];
        % this_3_States_matrix = variable_onset_struct.filled_play_bouts_3states;
        % 
        % for j=1:3
        %    this_3_States_matrix (variable_onset_struct.filled_play_bouts_3states(:,1)==j-1,1) = re_assignment(fn,j)-1;
        % end
        % all_hmm_lengths_3states = [all_hmm_lengths_3states; this_3_States_matrix];
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
         % all_variable_onsets_3states(:,:,total_number_of_hmm_3states+1:total_number_of_hmm_3states+n_hmm_3stsates,:)  = variable_onset_struct.beh_properties_onset_3states(re_assignment(fn,:),:,:,:) ;
        % all_variable_offsets_3states(:,:,total_number_of_hmm_3states+1:total_number_of_hmm_3states+n_hmm_3stsates,:)  = variable_onset_struct.beh_properties_offset_3states(re_assignment(fn,:),:,:,:) ;

        total_number_of_hmm = total_number_of_hmm+n_hmm
        % total_number_of_hmm_3states = total_number_of_hmm_3states+n_hmm_3stsates;
end
% plot3states_Variables(variable_onset_struct,re_assignment(fn,:))
psth_edges = variable_onset_struct.psth_edges;
%% now ploting variables (2 states)

[hmm_length_ordered, pb_order] = sort(all_hmm_lengths);
only_long = hmm_length_ordered>0

for variable_n=1:numel(variable_names)
    figure('units','normalized','outerposition',[0 0 .5 1]);
    colormap(1-gray)
    subplot(5,2,[1 3 5])
    matrix2plot = squeeze(all_variable_onsets(variable_n,pb_order(only_long),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(only_long)), matrix2plot)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(only_long))],'r')
    hold on
    plot(hmm_length_ordered(only_long),1:numel(hmm_length_ordered(only_long)),'r')
    axis xy
    title(variable_names{variable_n})

    subplot(5,2,[7 9])

    mean2plot = mean(matrix2plot);
    [~, ~, ci] = ttest(matrix2plot);

    fill([psth_edges fliplr(psth_edges )], [ci(1,:) fliplr(ci(2,:)) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none')
    hold on
    plot(psth_edges,mean2plot, 'k' )

    hold on
    plot([0 0],[min(min(ci)) max(max(ci))],'r')
    axis tight


    subplot(5,2,[1 3 5]+1)
    matrix2plot = squeeze(all_variable_offsets(variable_n,pb_order(only_long),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(only_long)), matrix2plot)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(only_long))],'r')
    hold on
    plot(-hmm_length_ordered(only_long),1:numel(hmm_length_ordered(only_long)),'r')
    axis xy
    title(variable_names{variable_n})

    subplot(5,2,[7 9]+1)

    mean2plot = mean(matrix2plot);
    [~, ~, ci] = ttest(matrix2plot);

    fill([psth_edges fliplr(psth_edges )], [ci(1,:) fliplr(ci(2,:)) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none')
    hold on
    plot(psth_edges,mean2plot, 'k' )

    hold on
    plot([0 0],[min(min(ci)) max(max(ci))],'r')
    axis tight


    pause(.1)


end

%% now ploting variables (3 states)


[hmm_length_ordered, pb_order] = sort(diff(all_hmm_lengths_3states(:,[2 3])'));
only_long = hmm_length_ordered'>=0
ordered_states = all_hmm_lengths_3states(pb_order,1);


for variable_n=1:numel(ALL_VARIABLE_NAMES)
    figure('units','normalized','outerposition',[0 0 1 1]);
    for hmm_n = 1:3
        index = ordered_states==hmm_n-1;
        colormap(1-gray)
        subplot(5,6,[1 7 13]+(hmm_n-1)*2)
        matrix2plot = squeeze(all_variable_onsets_3states(hmm_n,variable_n,pb_order,:));
        matrix2plot = matrix2plot(index,:);
        imagesc(psth_edges,1:numel(hmm_length_ordered(only_long & index)), matrix2plot)
        hold on
        plot([0 0],[1 numel(hmm_length_ordered(only_long & index))],'r')
        hold on
        plot(hmm_length_ordered(only_long & index),1:numel(hmm_length_ordered(only_long & index)),'r')
        axis xy
        title(ALL_VARIABLE_NAMES{variable_n})

        subplot(5,6,[19 25]+(hmm_n-1)*2)
        mean2plot = mean(matrix2plot);
        [~, ~, ci] = ttest(matrix2plot);

        fill([psth_edges fliplr(psth_edges )], [ci(1,:) fliplr(ci(2,:)) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none')
        hold on
        plot(psth_edges,mean2plot, 'k' )

        hold on
        plot([0 0],[min(min(ci)) max(max(ci))],'r')
        axis tight


        subplot(5,6,[1 7 13]+1+ (hmm_n-1)*2)
        matrix2plot = squeeze(all_variable_offsets_3states(hmm_n,variable_n,pb_order,:));
        matrix2plot = matrix2plot(index,:);
        imagesc(psth_edges,1:numel(hmm_length_ordered(only_long & index)), matrix2plot)
        hold on
        plot([0 0],[1 numel(hmm_length_ordered(only_long & index))],'r')
        hold on
        plot(-hmm_length_ordered(only_long & index),1:numel(hmm_length_ordered(only_long & index)),'r')
        axis xy
        title(ALL_VARIABLE_NAMES{variable_n})

        subplot(5,6,[19 25]+1+(hmm_n-1)*2)

        mean2plot = mean(matrix2plot);
        [~, ~, ci] = ttest(matrix2plot);

        fill([psth_edges fliplr(psth_edges )], [ci(1,:) fliplr(ci(2,:)) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none')
        hold on
        plot(psth_edges,mean2plot, 'k' )
        hold on
        plot([0 0],[min(min(ci)) max(max(ci))],'r')
        axis tight
    end
    pause(.1)
end
%% estiamating correlation

varirables2correlate = {'AnimalSpeed', 'PartnerSpeed'};
varialbes2comp = find(ismember(variable_names,varirables2correlate));
timepoints2use  = 25;
correlation_values_onset    = nan(size(all_variable_onsets, [2 3]));
correlation_values_offset   = nan(size(all_variable_onsets, [2 3]));
for trial = 1:numel(all_hmm_lengths)
    if mod(trial, 100)==0
        disp(trial)
    end

    for t=1:size(all_variable_onsets,3)-timepoints2use
        index2extract =t:(t+timepoints2use);
        y1 = squeeze(all_variable_onsets(varialbes2comp(1),trial,index2extract));
        y2 = squeeze(all_variable_onsets(varialbes2comp(2),trial,index2extract));

        [c,p] = corr(y1, y2);
        correlation_values_onset(trial,t) = c;

         y1 = squeeze(all_variable_offsets(varialbes2comp(1),trial,index2extract));
        y2 = squeeze(all_variable_offsets(varialbes2comp(2),trial,index2extract));

        [c,p] = corr(y1, y2);
        correlation_values_offset(trial,t) = c;
    end
end
save('correlation_speed','correlation_values_onset','correlation_values_offset','timepoints2use', 'varirables2correlate','psth_edges')
%% plot correlation

[hmm_length_ordered, pb_order] = sort(all_hmm_lengths);
only_long = hmm_length_ordered>0
figure('units','normalized','outerposition',[0 0 .5 1]);
colormap(1-gray)
subplot(5,2,[1 3 5])
matrix2plot = correlation_values_onset(pb_order(only_long),:);
imagesc(psth_edges,1:numel(hmm_length_ordered(only_long)), matrix2plot)
hold on
plot([0 0],[1 numel(hmm_length_ordered(only_long))],'r')
hold on
plot(hmm_length_ordered(only_long),1:numel(hmm_length_ordered(only_long)),'r')
axis xy
title('Correlation onset')

subplot(5,2,[7 9])

mean2plot = mean(matrix2plot,'omitmissing');
[~, ~, ci] = ttest(matrix2plot);
no_nan = ~any(isnan(ci));

fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,no_nan) fliplr(ci(2,no_nan)) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none')
hold on
plot(psth_edges,mean2plot, 'k' )

hold on
plot([0 0],[min(min(ci)) max(max(ci))],'r')
axis tight


subplot(5,2,[1 3 5]+1)
matrix2plot = correlation_values_offset(pb_order(only_long),:);
imagesc(psth_edges,1:numel(hmm_length_ordered(only_long)), matrix2plot)
hold on
plot([0 0],[1 numel(hmm_length_ordered(only_long))],'r')
hold on
plot(-hmm_length_ordered(only_long),1:numel(hmm_length_ordered(only_long)),'r')
axis xy
title('Correlation offset')
subplot(5,2,[7 9]+1)

mean2plot = mean(matrix2plot,'omitmissing');
[~, ~, ci] = ttest(matrix2plot);
no_nan = ~any(isnan(ci));

fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,no_nan) fliplr(ci(2,no_nan)) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none')
hold on
plot(psth_edges,mean2plot, 'k' )

hold on
plot([0 0],[min(min(ci)) max(max(ci))],'r')
axis tight


pause(.1)
%% estimate correaltion and length
onset_index =psth_edges>-.35 & psth_edges<=.35 ;
offset_index = psth_edges>=-.4 & psth_edges<=.0 ;
mean_cor = [mean(correlation_values_onset(:,onset_index),2) mean(correlation_values_offset(:,offset_index),2) ];

figure
subplot(1,2,1)
plot(abs(mean_cor(:,1)),all_hmm_lengths, '.')
subplot(1,2,2)
plot(abs(mean_cor(:,2)),all_hmm_lengths, '.')
no_nan = ~any(isnan(mean_cor),2) & ~isnan(all_hmm_lengths);
[c,p] = corr(abs(mean_cor(no_nan,1)),all_hmm_lengths(no_nan));


%% prepare data for classification
   hmm_type = any(is_this_hmm  & is_there_play_beh,2 );


hmm_data_mean = nan(numel(all_hmm_lengths), size(all_variable_onsets,1)+2);
hmm_data_all = [];
bout_index = [];

   hmm_n = 1;
  
   for hmm_n=1:numel(all_hmm_lengths)
        hmm_indexes = is_this_hmm(hmm_n,:)==1;
        bout_index = repmat(hmm_n,size(hmm_indexes));
        bout_index=bout_index(hmm_indexes);
        hmm_data_all = [hmm_data_all;[squeeze(all_variable_onsets(:,hmm_n,hmm_indexes))' bout_index' repmat([all_hmm_lengths(hmm_n) hmm_type(hmm_n) ], sum(hmm_indexes), 1) is_there_play_beh(hmm_n,hmm_indexes)']];
        hmm_data_mean(hmm_n,:) = [mean(squeeze(all_variable_onsets(:,hmm_n,hmm_indexes)),2)' all_hmm_lengths(hmm_n) hmm_type(hmm_n)] ;
   end

 hmm_data_all = array2table(hmm_data_all)  ;
 hmm_data_all.Properties.VariableNames = strsplit(num2str(1:size(hmm_data_all,2)),' ');
 hmm_data_all.Properties.VariableNames{end}     = 'PlayBool';
 hmm_data_all.Properties.VariableNames{end-1}   = 'StateType';
 hmm_data_all.Properties.VariableNames{end-2}   = 'StateLength';
 hmm_data_all.Properties.VariableNames{end-3}   = 'StateNum';
%% save (or loead)

 save('SVM_play_predict.mat','SVM_play_predict')
list_of_Var = SVM_play_predict.RequiredVariables;

table2predict = hmm_data_all(:, ismember(hmm_data_all.Properties.VariableNames,list_of_Var));
 [yfit,scores] = SVM_play_predict.predictFcn(table2predict);

 %%

score_values = nan(size(is_this_hmm));



 for hmm_n=1:numel(all_hmm_lengths)
 
    hmm_indexes = is_this_hmm(hmm_n,:)==1;
    score_values(hmm_n,hmm_indexes) = scores(hmm_data_all.StateNum==hmm_n,1);
 end



 
 
 %%
figure

imagesc(psth_edges, 1:numel(pb_order), score_values(pb_order,:))
axis xy
xlim([-1 1])
%%
positve_scores = score_values;
% positve_scores(positve_scores<0) = NaN;

state_params = [mean(positve_scores,2, 'omitmissing') median(positve_scores,2, 'omitmissing') max(positve_scores,[],2, 'omitmissing') min(positve_scores,[],2, 'omitmissing')];


% state_params_zscored  = [zscore(mean(score_values,2, 'omitmissing'))  zscore(median(score_values,2, 'omitmissing')) zscore(max(score_values,[],2, 'omitmissing') ) zscore(min(score_values,[],2, 'omitmissing'))];

[coeff,score_pca,latent,tsquared,explained,mu]  = pca(state_params_zscored); 

figure
ax = subplot(1,2,1);
plot(score_pca(hmm_type==1,1), score_pca(hmm_type==1,2), 'r.')
data = score_pca(hmm_type==1,[1 2]);
plot_2d_percentile_contours(data, 100*[.5 .78 .9], ax, 'r')
hold on
plot(score_pca(hmm_type==0,1), score_pca(hmm_type==0,2), 'k.')
data = score_pca(hmm_type==0,[1 2]);
plot_2d_percentile_contours(data, 100*[.5 .78 .9], ax, 'k')
%%

figure

plot(state_params(hmm_type==1,4), state_params(hmm_type==1,3), 'r.')

hold on

hold on
plot(state_params(hmm_type==0,4), state_params(hmm_type==0,3), 'k.')

%%
figure

histogram(score_pca(hmm_type==0,1), -2:.1:6,'Normalization','percentage' )
hold on
histogram(score_pca(hmm_type==1,1), -2:.1:6,'Normalization','percentage' )


subplot(1,2,2)
histogram(state_params(hmm_type==0,2), -2:.1:6)
hold on
histogram(state_params(hmm_type==1,2), -2:.1:6)
%%
tbl = table(state_params(:,1), state_params(:,2),state_params(:,3),state_params(:,4), hmm_type, ...
    'VariableNames', {'Mean','Median','Max','Min','Hmmtype'});

% Fit logistic regression:
mdl = fitglm(tbl, 'Hmmtype ~ Mean + Median + Max +  Min', 'Distribution', 'binomial');

probs = predict(mdl, tbl(:,1:4));
class = round(probs);


tp = sum(class == 1 & hmm_type==1)/sum( hmm_type==1);
fn = sum(class == 0 & hmm_type==1)/sum( hmm_type==1);
tn = sum(class == 0 & hmm_type==0)/sum( hmm_type==0);
fp = sum(class == 1 & hmm_type==0)/sum( hmm_type==0);

 %%

 figure
 histogram(scores(hmm_data_all.StateType==1,1))
 hold on
 histogram(scores(hmm_data_all.StateType==0,1))
 hold on


%%


  



%%

figure
range2average = psth_edges>=0  & psth_edges<0.2;
mean_correaltion = mean((correlation_values_onset(:,range2average)).^2,2)
no_nan = ~isnan(mean_correaltion)
plot(mean_correaltion, all_hmm_lengths,'.')
glm_fit = fitglm(mean_correaltion, all_hmm_lengths, 'Distribution','gamma', 'Link','log');
plot(glm_fit)
xlabel('r2')

ylabel('HMM length (s)')
[c,p] = corr(all_hmm_lengths(no_nan),mean_correaltion(no_nan), 'type','Spearman')
%% testing granger causality


varialbes2comp = [1 5];

optimal_lags = nan(numel(all_hmm_lengths),1);

for trial = 1:numel(all_hmm_lengths)
index2extract = psth_edges>-.1 & psth_edges<=all_hmm_lengths(trial);
y1 = squeeze(all_variable_onsets(varialbes2comp(1),trial,index2extract));
y2 = squeeze(all_variable_onsets(varialbes2comp(2),trial,index2extract));
Y = [y1, y2];

maxLags = min(10, round(numel(y1)/4));
aic = zeros(maxLags, 1);

for p = 1:maxLags
    Mdl = varm(2, p);
    EstMdl = estimate(Mdl, Y);
    summ_mdl = summarize(EstMdl);
    aic(p)=summ_mdl.AIC;
end



% Choose lag with minimum AIC
[~, optimalLag] = min(aic);


optimal_lags(trial) = optimalLag;
end
%% estimating GC for onset  (long)
optimalLag = 4;

timepoints2use = 100;
granger_matrix_a2p = nan(size(all_variable_onsets, [2 3]));
granger_matrix_p2a = nan(size(all_variable_onsets, [2 3]));
warning('off','all')
for trial = 12:numel(all_hmm_lengths)
    disp(trial)

    for t=900:size(all_variable_onsets,3)-timepoints2use
        index2extract =t:(t+timepoints2use);
        y1 = squeeze(all_variable_onsets(varialbes2comp(1),trial,index2extract));
        y2 = squeeze(all_variable_onsets(varialbes2comp(2),trial,index2extract));
        Y = [y1, y2];

        if max(sum(y1==0), sum(y2==0))<25

        Mdl = varm(2, optimalLag);
        EstMdl = estimate(Mdl, Y);
    try
        [h,pValue, stat] = gctest(Y(:,1), Y(:,2),  NumLags= optimalLag);
        granger_matrix_a2p(trial,t+round(timepoints2use/2)) = stat;

        [h,pValue, stat] = gctest(Y(:,2), Y(:,1), NumLags=optimalLag);
        granger_matrix_p2a(trial,t+round(timepoints2use/2)) = stat;
    catch
    end
        end
    end
end
save('granger_matrixes', 'granger_matrix_p2a', 'granger_matrix_a2p','optimalLag','timepoints2use')
%% now ploting (need to be redone)



%%


x_lim = [-2 2];
is_there_play = any(is_this_hmm & is_there_play_beh,2);
there_is_no_play = ~any(is_this_hmm & is_there_play_beh,2);

[hmm_length_ordered, pb_order] = sort(all_hmm_lengths);
only_long = hmm_length_ordered>0;


for variable_n=1:numel(variable_names)
    figure('units','normalized','outerposition',[0 0 .5 1]);
    colormap(1-gray)
    subplot(5,2,1)
    matrix2plot_withplay = squeeze(all_variable_onsets(variable_n,pb_order(is_there_play(pb_order)),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(is_there_play(pb_order))), matrix2plot_withplay)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(is_there_play(pb_order)))],'r')
    hold on
    plot(hmm_length_ordered(is_there_play(pb_order)),1:numel(hmm_length_ordered(is_there_play(pb_order))),'r')
    axis xy
    yticks([])
    ylabel('With Play')
    xlim(x_lim)
    title(variable_names{variable_n})

    subplot(5,2,3)
    matrix2plot_withoutplay = squeeze(all_variable_onsets(variable_n,pb_order(there_is_no_play(pb_order)),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(there_is_no_play(pb_order))), matrix2plot_withoutplay)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(there_is_no_play(pb_order)))],'r')
    hold on
    plot(hmm_length_ordered(there_is_no_play(pb_order)),1:numel(hmm_length_ordered(there_is_no_play(pb_order))),'r')
    axis xy
    xlim(x_lim)
     yticks([])
    ylabel('Without Play')


    subplot(5,2,[5 7 9])
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
      legend({ 'Without Play','Play'})
      axis tight
    xlim(x_lim)
  
    subplot(5,2,2)
    matrix2plot_withplay = squeeze(all_variable_offsets(variable_n,pb_order(is_there_play(pb_order)),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(is_there_play(pb_order))), matrix2plot_withplay)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(is_there_play(pb_order)))],'r')
    hold on
    plot(-hmm_length_ordered(is_there_play(pb_order)),1:numel(hmm_length_ordered(is_there_play(pb_order))),'r')
    axis xy
    xlim(x_lim)
     yticks([])
    ylabel('With Play')
    title(variable_names{variable_n})

    subplot(5,2,4)
    matrix2plot_withoutplay = squeeze(all_variable_offsets(variable_n,pb_order(there_is_no_play(pb_order)),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(there_is_no_play(pb_order))), matrix2plot_withoutplay)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(there_is_no_play(pb_order)))],'r')
    hold on
    plot(-hmm_length_ordered(there_is_no_play(pb_order)),1:numel(hmm_length_ordered(there_is_no_play(pb_order))),'r')
    axis xy
    xlim(x_lim)
     yticks([])
    ylabel('Without Play')


    subplot(5,2,[5 7 9]+1)
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
    legend({ 'Without Play','Play'})
      axis tight
    xlim(x_lim)

    pause(.1)
    % saveas(gcf, [variable_names{variable_n}, '.jpg'])
end


%% same for correlation

figure('units','normalized','outerposition',[0 0 .5 1]);
colormap(1-gray)
subplot(5,2,[1 3 5])
x_lim = [-2 4];
is_there_play  =  any(is_this_hmm & is_there_play_beh,2) & all_hmm_lengths>0.5;
there_is_no_play =~any(is_this_hmm & is_there_play_beh,2) & all_hmm_lengths>0.5

figure('units','normalized','outerposition',[0 0 .5 1]);
    colormap(1-gray)
    subplot(5,2,1)
    matrix2plot_withplay = squeeze(correlation_values_onset(pb_order(is_there_play(pb_order)),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(is_there_play(pb_order))), matrix2plot_withplay)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(is_there_play(pb_order)))],'r')
    hold on
    plot(hmm_length_ordered(is_there_play(pb_order)),1:numel(hmm_length_ordered(is_there_play(pb_order))),'r')
    axis xy
    yticks([])
    ylabel('With Play')
    xlim(x_lim)
    title('HMM onset correlation ')

    subplot(5,2,3)
    matrix2plot_withoutplay = squeeze(correlation_values_onset(pb_order(there_is_no_play(pb_order)),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(there_is_no_play(pb_order))), matrix2plot_withoutplay)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(there_is_no_play(pb_order)))],'r')
    hold on
    plot(hmm_length_ordered(there_is_no_play(pb_order)),1:numel(hmm_length_ordered(there_is_no_play(pb_order))),'r')
    axis xy
    xlim(x_lim)
     yticks([])
    ylabel('Without Play')


    subplot(5,2,[5 7 9])
    mean2plot = mean(matrix2plot_withoutplay, 'omitmissing');
    [~, ~, ci] = ttest(matrix2plot_withoutplay);
    no_nan = ~any(isnan(ci));
    fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,(no_nan)) fliplr(ci(2,(no_nan))) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    hold on
    plot(psth_edges,mean2plot, 'k' )
    mean2plot = mean(matrix2plot_withplay, 'omitmissing');
    [~, ~, ci] = ttest(matrix2plot_withplay);
    no_nan = ~any(isnan(ci));
    fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,(no_nan)) fliplr(ci(2,(no_nan))) ], 'r', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    hold on
    plot(psth_edges,mean2plot, 'r' )
    y_lim = ylim;
    hold on
    plot([0 0],y_lim,'b', 'HandleVisibility','off')
      legend({ 'Without Play','Play'})
      axis tight
    xlim(x_lim)
  
    subplot(5,2,2)
    matrix2plot_withplay = squeeze(correlation_values_offset(pb_order(is_there_play(pb_order)),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(is_there_play(pb_order))), matrix2plot_withplay)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(is_there_play(pb_order)))],'r')
    hold on
    plot(-hmm_length_ordered(is_there_play(pb_order)),1:numel(hmm_length_ordered(is_there_play(pb_order))),'r')
    axis xy
    xlim(x_lim)
     yticks([])
    ylabel('With Play')
    title('HMM offset correlation ')

    subplot(5,2,4)
    matrix2plot_withoutplay = squeeze(correlation_values_offset(pb_order(there_is_no_play(pb_order)),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(there_is_no_play(pb_order))), matrix2plot_withoutplay)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(there_is_no_play(pb_order)))],'r')
    hold on
    plot(-hmm_length_ordered(there_is_no_play(pb_order)),1:numel(hmm_length_ordered(there_is_no_play(pb_order))),'r')
    axis xy
    xlim(x_lim)
     yticks([])
    ylabel('Without Play')


    subplot(5,2,[5 7 9]+1)
    mean2plot = mean(matrix2plot_withoutplay, 'omitmissing');
    [~, ~, ci] = ttest(matrix2plot_withoutplay);
    no_nan = ~any(isnan(ci));
    fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,(no_nan)) fliplr(ci(2,(no_nan))) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    hold on
    plot(psth_edges,mean2plot, 'k' )
    mean2plot = mean(matrix2plot_withplay, 'omitmissing');
    [~, ~, ci] = ttest(matrix2plot_withplay);
     no_nan = ~any(isnan(ci));
    fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,(no_nan)) fliplr(ci(2,(no_nan))) ], 'r', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    hold on
    plot(psth_edges,mean2plot, 'r' )
    y_lim = ylim;
    hold on
    plot([0 0],y_lim,'b', 'HandleVisibility','off')
    legend({ 'Without Play','Play'})
      axis tight
    xlim(x_lim)

%% same plots (play and no play) but mathcing lenghts


is_there_play  =  any(is_this_hmm & is_there_play_beh,2);
there_is_no_play =~any(is_this_hmm & is_there_play_beh,2);

play_lengths    = all_hmm_lengths(is_there_play);
noplay_lengths  = all_hmm_lengths(there_is_no_play);

Cost = (play_lengths-noplay_lengths').^2;

%%
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


%% 
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

%% now ploting


x_lim = [-2 4];
is_there_play  = is_there_play & all_hmm_lengths>0;
there_is_no_play = there_is_no_play & all_hmm_lengths>0;

[hmm_length_ordered, pb_order] = sort(all_hmm_lengths);
%%


for variable_n=1:numel(variable_names)
    figure('units','normalized','outerposition',[0 0 .5 1]);
    colormap(1-gray)
    subplot(5,2,1)
    matrix2plot_withplay = squeeze(all_variable_onsets(variable_n,pb_order(is_there_play(pb_order)),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(is_there_play(pb_order))), matrix2plot_withplay)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(is_there_play(pb_order)))],'r')
    hold on
    plot(hmm_length_ordered(is_there_play(pb_order)),1:numel(hmm_length_ordered(is_there_play(pb_order))),'r')
    axis xy
    yticks([])
    ylabel('With Play')
    xlim(x_lim)
    title(variable_names{variable_n})

    subplot(5,2,3)
    matrix2plot_withoutplay = squeeze(all_variable_onsets(variable_n,pb_order(there_is_no_play(pb_order)),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(there_is_no_play(pb_order))), matrix2plot_withoutplay)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(there_is_no_play(pb_order)))],'r')
    hold on
    plot(hmm_length_ordered(there_is_no_play(pb_order)),1:numel(hmm_length_ordered(there_is_no_play(pb_order))),'r')
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
        lengths_to_include = hmm_length_ordered(there_is_no_play(pb_order))>=psth_edges(j);
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
        lengths_to_include = hmm_length_ordered(is_there_play(pb_order))>=psth_edges(j);
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
    imagesc(psth_edges,1:numel(hmm_length_ordered(is_there_play(pb_order))), matrix2plot_withplay)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(is_there_play(pb_order)))],'r')
    hold on
    plot(-hmm_length_ordered(is_there_play(pb_order)),1:numel(hmm_length_ordered(is_there_play(pb_order))),'r')
    axis xy
    xlim(x_lim)
     yticks([])
    ylabel('With Play')
    title(variable_names{variable_n})

    subplot(5,2,4)
    matrix2plot_withoutplay = squeeze(all_variable_offsets(variable_n,pb_order(there_is_no_play(pb_order)),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(there_is_no_play(pb_order))), matrix2plot_withoutplay)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(there_is_no_play(pb_order)))],'r')
    hold on
    plot(-hmm_length_ordered(there_is_no_play(pb_order)),1:numel(hmm_length_ordered(there_is_no_play(pb_order))),'r')
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
        lengths_to_include = hmm_length_ordered(there_is_no_play(pb_order))>=-psth_edges(j);
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
        lengths_to_include = hmm_length_ordered(is_there_play(pb_order))>=-psth_edges(j);
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
    saveas(gcf, [variable_names{variable_n}, '.jpg'])
end
%%
figure('units','normalized','outerposition',[0 0 .5 1]);
    colormap(1-gray)
    subplot(5,2,1)
    matrix2plot_withplay = squeeze(correlation_values_onset(pb_order(is_there_play(pb_order)),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(is_there_play(pb_order))), matrix2plot_withplay)
    clim([-1 1])
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(is_there_play(pb_order)))],'r')
    hold on
    plot(hmm_length_ordered(is_there_play(pb_order)),1:numel(hmm_length_ordered(is_there_play(pb_order))),'r')
    axis xy
    yticks([])
    ylabel('With Play')
    xlim(x_lim)
    title('Correlation onset')

    subplot(5,2,3)
    matrix2plot_withoutplay = squeeze(correlation_values_onset(pb_order(there_is_no_play(pb_order)),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(there_is_no_play(pb_order))), matrix2plot_withoutplay)
    clim([-1 1])

    hold on
    plot([0 0],[1 numel(hmm_length_ordered(there_is_no_play(pb_order)))],'r')
    hold on
    plot(hmm_length_ordered(there_is_no_play(pb_order)),1:numel(hmm_length_ordered(there_is_no_play(pb_order))),'r')
    axis xy
    xlim(x_lim)
     yticks([])
    ylabel('Without Play')


    subplot(5,2,5)
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
    plot(psth_edges,mean2plot, 'r' )
    y_lim = ylim;
    hold on
    plot([0 0],y_lim,'b', 'HandleVisibility','off')
     
      axis tight
    xlim(x_lim)


    subplot(5,2,[7 9])
    mean2plot      = mean(matrix2plot_withoutplay, 'omitmissing');
    [~, ~, ci] = ttest(matrix2plot_withoutplay);
    for j=find(psth_edges>=0)
        lengths_to_include = hmm_length_ordered(there_is_no_play(pb_order))>=psth_edges(j) & ~isnan(matrix2plot_withoutplay(:,j));
        mean2plot(j) = mean(matrix2plot_withoutplay(lengths_to_include,j));
        ci(:,j) = mean2plot(j) + 1.96*std(matrix2plot_withoutplay(lengths_to_include,j))*[-1 1]/sqrt(sum(lengths_to_include));
    end
    no_nan = ~any(isnan(ci));
    fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,no_nan) fliplr(ci(2,no_nan)) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    hold on
    plot(psth_edges,mean2plot, 'k' )

    mean2plot      = mean(matrix2plot_withplay, 'omitmissing');
    [~, ~, ci] = ttest(matrix2plot_withplay);
    for j=find(psth_edges>=0)
        lengths_to_include = hmm_length_ordered(is_there_play(pb_order))>=psth_edges(j) & ~isnan(matrix2plot_withplay(:,j));
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
     matrix2plot_withplay = squeeze(correlation_values_offset(pb_order(is_there_play(pb_order)),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(is_there_play(pb_order))), matrix2plot_withplay)
    clim([-1 1])
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(is_there_play(pb_order)))],'r')
    hold on
    plot(-hmm_length_ordered(is_there_play(pb_order)),1:numel(hmm_length_ordered(is_there_play(pb_order))),'r')
    axis xy
    xlim(x_lim)
     yticks([])
    ylabel('With Play')
    title('Correlation offset')

    subplot(5,2,4)
    matrix2plot_withoutplay = squeeze(correlation_values_offset(pb_order(there_is_no_play(pb_order)),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(there_is_no_play(pb_order))), matrix2plot_withoutplay)   
    clim([-1 1])
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(there_is_no_play(pb_order)))],'r')
    hold on
    plot(-hmm_length_ordered(there_is_no_play(pb_order)),1:numel(hmm_length_ordered(there_is_no_play(pb_order))),'r')
    axis xy
    xlim(x_lim)
     yticks([])
    ylabel('Without Play')


    subplot(5,2,6)
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
    plot(psth_edges,mean2plot, 'r' )
    y_lim = ylim;
    hold on
    plot([0 0],y_lim,'b', 'HandleVisibility','off')
     
      axis tight
    xlim(x_lim)
    
    y_lim = ylim;
    hold on
    plot([0 0],y_lim,'b', 'HandleVisibility','off')
      axis tight
    xlim(x_lim)

    subplot(5,2,[8 10])
    mean2plot      = mean(matrix2plot_withoutplay, 'omitmissing');
    [~, ~, ci] = ttest(matrix2plot_withoutplay);
    for j=find(psth_edges<=0)
        lengths_to_include = hmm_length_ordered(there_is_no_play(pb_order))>=-psth_edges(j) & ~isnan(matrix2plot_withoutplay(:,j));
        mean2plot(j) = mean(matrix2plot_withoutplay(lengths_to_include,j));
        ci(:,j) = mean2plot(j) + 1.96*std(matrix2plot_withoutplay(lengths_to_include,j))*[-1 1]/sqrt(sum(lengths_to_include));
    end
    no_nan = ~any(isnan(ci));
     fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,no_nan) fliplr(ci(2,no_nan)) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    hold on
    plot(psth_edges,mean2plot, 'k' )

     mean2plot      = mean(matrix2plot_withplay, 'omitmissing');
    [~, ~, ci] = ttest(matrix2plot_withplay);
    for j=find(psth_edges<=0)
        lengths_to_include = hmm_length_ordered(is_there_play(pb_order))>=-psth_edges(j) & ~isnan(matrix2plot_withplay(:,j));
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

    %%
figure('units','normalized','outerposition',[0 0 .5 1]);
    colormap(1-gray)
    subplot(5,2,1)
   
    imagesc(psth_edges,1:numel(hmm_length_ordered(is_there_play(pb_order))), matrix2plot_withplay)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(is_there_play(pb_order)))],'r')
    hold on
    plot(hmm_length_ordered(is_there_play(pb_order)),1:numel(hmm_length_ordered(is_there_play(pb_order))),'r')
    axis xy
    yticks([])
    ylabel('With Play')
    xlim(x_lim)
    title('HMM onset correlation ')

    subplot(5,2,3)
    imagesc(psth_edges,1:numel(hmm_length_ordered(there_is_no_play(pb_order))), matrix2plot_withoutplay)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(there_is_no_play(pb_order)))],'r')
    hold on
    plot(hmm_length_ordered(there_is_no_play(pb_order)),1:numel(hmm_length_ordered(there_is_no_play(pb_order))),'r')
    axis xy
    xlim(x_lim)
     yticks([])
    ylabel('Without Play')


    subplot(5,2,[5 7 9])
    mean2plot = mean(matrix2plot_withoutplay, 'omitmissing');
    [~, ~, ci] = ttest(matrix2plot_withoutplay);
    no_nan = ~any(isnan(ci));
    fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,(no_nan)) fliplr(ci(2,(no_nan))) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    hold on
    plot(psth_edges,mean2plot, 'k' )
    mean2plot = mean(matrix2plot_withplay, 'omitmissing');
    [~, ~, ci] = ttest(matrix2plot_withplay);
    no_nan = ~any(isnan(ci));
    fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,(no_nan)) fliplr(ci(2,(no_nan))) ], 'r', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    hold on
    plot(psth_edges,mean2plot, 'r' )
    y_lim = ylim;
    hold on
    plot([0 0],y_lim,'b', 'HandleVisibility','off')
      legend({ 'Without Play','Play'})
      axis tight
    xlim(x_lim)
  
    subplot(5,2,2)
    matrix2plot_withplay = squeeze(correlation_values_offset(pb_order(is_there_play(pb_order)),:));
    imagesc(psth_edges,1:numel(hmm_length_ordered(is_there_play(pb_order))), matrix2plot_withplay)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(is_there_play(pb_order)))],'r')
    hold on
    plot(-hmm_length_ordered(is_there_play(pb_order)),1:numel(hmm_length_ordered(is_there_play(pb_order))),'r')
    axis xy
    xlim(x_lim)
     yticks([])
    ylabel('With Play')
    title('HMM offset correlation ')

    subplot(5,2,4)
    imagesc(psth_edges,1:numel(hmm_length_ordered(there_is_no_play(pb_order))), matrix2plot_withoutplay)
    hold on
    plot([0 0],[1 numel(hmm_length_ordered(there_is_no_play(pb_order)))],'r')
    hold on
    plot(-hmm_length_ordered(there_is_no_play(pb_order)),1:numel(hmm_length_ordered(there_is_no_play(pb_order))),'r')
    axis xy
    xlim(x_lim)
     yticks([])
    ylabel('Without Play')


    subplot(5,2,[5 7 9]+1)
    mean2plot = mean(matrix2plot_withoutplay, 'omitmissing');
    [~, ~, ci] = ttest(matrix2plot_withoutplay);
    no_nan = ~any(isnan(ci));
    fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,(no_nan)) fliplr(ci(2,(no_nan))) ], 'k', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    hold on
    plot(psth_edges,mean2plot, 'k' )
    mean2plot = mean(matrix2plot_withplay, 'omitmissing');
    [~, ~, ci] = ttest(matrix2plot_withplay);
     no_nan = ~any(isnan(ci));
    fill([psth_edges(no_nan) fliplr(psth_edges(no_nan) )], [ci(1,(no_nan)) fliplr(ci(2,(no_nan))) ], 'r', 'FaceAlpha',.5, 'EdgeColor','none', 'HandleVisibility','off')
    hold on
    plot(psth_edges,mean2plot, 'r' )
    y_lim = ylim;
    hold on
    plot([0 0],y_lim,'b', 'HandleVisibility','off')
    legend({ 'Without Play','Play'})
      axis tight
    xlim(x_lim)

    %% ploting properties for each behavior


%%
    bt_1 = 6;
    bt_2 = 17;
    bt_3 = 4;
    vn = 14;
variable_names{vn}

behavior_type_list{bt_2}


figure
this_var_matrix = squeeze(all_variable_onsets(vn,:,:));
this_behavior_matrix_1 = squeeze(merged_behaviors_onset(bt_1,:,:));
histogram(this_var_matrix(this_behavior_matrix_1==1 & current_hmm),0:0.05:.8, 'Normalization','percentage')
this_behavior_matrix_2 = squeeze(merged_behaviors_onset(bt_2,:,:));
hold on
histogram(this_var_matrix(this_behavior_matrix_2==1 & current_hmm),0:0.05:.8, 'Normalization','percentage')
this_behavior_matrix_3 = squeeze(merged_behaviors_onset(bt_3,:,:));
hold on
histogram(this_var_matrix(this_behavior_matrix_3==1 & current_hmm),0:0.05:.8, 'Normalization','percentage')
hold on
histogram(this_var_matrix(this_behavior_matrix_2==1 & ~current_hmm),0:0.05:.8, 'Normalization','percentage')
legend(behavior_type_list([bt_1, bt_2,bt_3]))