saving_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Codes\Play bout codes';
behavior_data = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Behavior backups';
save_bool = false;
current_folder = cd;
cd(behavior_data)
behavior_files = dir('*.txt');

beh_meta_data = cell(numel(behavior_files),3);

for bf = 1:numel(behavior_files)
    this_file_metadata = strsplit(behavior_files(bf).name, ' ');
    this_file_metadata{3}= strsplit(this_file_metadata{3}, '.');
    this_file_metadata{3} = this_file_metadata{3}{1};
    this_file_metadata{2} = str2double(this_file_metadata{2});

    beh_meta_data(bf,:) =this_file_metadata;
end

beh_meta_data = cell2table(beh_meta_data);
beh_meta_data.Properties.VariableNames = {'Animal','Session','Implanted'};
lag_time = 20;
beh_bin = 0.01;
conv_length = 1;
behavior_psth_hist_Range = [-5 5];
wraped_bins = 50;


hist_range = [0 20];
hist_bins = .1;
hist_edges = hist_range(1):hist_bins:hist_range(2);
hist_edges_centers = .5*(hist_edges(1:end-1)+hist_edges(2:end));


cc_matrix           = nan(numel(behavior_files),2*(lag_time/beh_bin) +1 );
inter_bout_matrix   = nan(numel(behavior_files),numel(hist_edges_centers));
bout_length_matrix  = nan(numel(behavior_files),numel(hist_edges_centers));

renage_around = 1.25;


play_Structure = [];
play_Structure.behavior_matrix  = [];
play_Structure.behavior_list    = [];

%%
restrain2morethan1beh = false;
ALL_behavior_matrices   = [];
play_behaviors      = {'Pounce', 'CC','Boxing', 'Evasion','Pin', 'CB', 'CD', 'Escape'};
beh_matrix_range    = [-20 20];
beh_matrix_bin      = 0.01;
beh_matrix_indexes  = beh_matrix_range(1):beh_matrix_bin:beh_matrix_range(2);


all_behavior_list = [];
for bf = 1:numel(behavior_files)
    Behavior =   readtable(behavior_files(bf).name);
    Behavior(:,2) = [];
    Behavior.Properties.VariableNames = {'Animal', 'Start', 'End', 'Length', 'Type'};
    Behavior.Type2 = Behavior.Type;
    Behavior.Type2(ismember(Behavior.Type2, {'Pounce_A', 'Pounce_B'})) = {'Pounce'}; %% Merging behaviors to Type2
    Behavior.Type2(ismember(Behavior.Type2, {'Pounce_Ai', 'Pounce_Bi'})) = {'PounceI'};
    Behavior.Type2(ismember( Behavior.Type2,'')) = {'Other'};
    Behavior(ismember(Behavior.Animal, 'Reversal'),:) = [];

    animal_types = unique(Behavior.Animal);
    repeated_animal = beh_meta_data.Implanted(bf); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    partner_names = animal_types;
    partner_names(ismember(partner_names, repeated_animal)) = [];

    partner_names(ismember(partner_names, 'Session_structure')) = [];
    animal_types(ismember(animal_types, 'Session_structure')) = [];
    new_behavior = [];

    sessions_structure =  Behavior(ismember(Behavior.Animal, 'Session_structure'),:);
    sessions_structure(ismember(sessions_structure.Type, 'Tickling'),:) = [];

    for sn=1:size(sessions_structure,1)
        new_behavior = [new_behavior;Behavior(Behavior.Start>=sessions_structure.Start(sn) & Behavior.End<=sessions_structure.End(sn),:)];
    end

    Behavior = new_behavior;

    animal_sessions =sessions_structure{:, {'Start','End'}};

    main_partner = nan(size(animal_sessions,1),numel(partner_names));

    for  sn=1:size(sessions_structure,1)
        for pt=1:numel(partner_names)
            main_partner(sn,pt) = sum(ismember(Behavior.Animal(Behavior.Start>=sessions_structure.Start(sn) & Behavior.End<=sessions_structure.End(sn),:),partner_names{pt}));
        end
    end

    [~, loc]  =max(main_partner,[],2);
    partner_per_session = partner_names(loc);

    for pt=1:numel(partner_names)
        Behavior(Behavior.Start>=sessions_structure.Start(pt) & Behavior.End<=sessions_structure.End(pt) & ~ismember(Behavior.Animal, partner_per_session{pt}),:) = [];
    end

    new_behavior = [];
    for sn=1:size(sessions_structure,1)
        new_behavior = [new_behavior;Behavior(Behavior.Start>=sessions_structure.Start(sn) & Behavior.End<=sessions_structure.End(sn),:)];
    end

    Behavior = new_behavior;



 

    Behavior(ismember(Behavior.Type,'Partners session'),:)=[];

  

    config.Behavior = Behavior;
    config.repeated_animal =repeated_animal;
    config.animal_types =   animal_types        ;
    config.play_behaviors =play_behaviors      ;
    config.beh_bin =beh_bin             ;
    config.conv_length =conv_length;       
    config.behavior_window  = 0;

        [play_bouts_table] = play_bout(config) ;


        all_beh_type = unique(Behavior.Type);
        behavior_matrix = zeros(numel(all_beh_type),size(play_bouts_table,1),wraped_bins*3);
        

        for pb = 1:size(play_bouts_table,1)
            pb_start    = play_bouts_table(pb,1);
            pb_end      = play_bouts_table(pb,2);
            
            t1          = pb_start+behavior_psth_hist_Range(1);
            t2          = pb_start;
            t3          = pb_end;
            t4          = pb_end+behavior_psth_hist_Range(2);

            time_centers_1 = linspace(t1,t2,wraped_bins+1);
            time_centers_2 = linspace(t2,t3,wraped_bins+1);
            time_centers_3 = linspace(t3,t4,wraped_bins+1);

            all_time_centers = [time_centers_1(1:end-1),time_centers_2(1:end-1),time_centers_3(1:end-1)];


            for bn=1:numel(all_beh_type)

                all_this_behaviors = find(ismember(Behavior.Type, all_beh_type{bn}))';

                for bn_n = 1:numel(all_this_behaviors)
                index_with_behavior = Behavior.Start(all_this_behaviors(bn_n))<=all_time_centers & all_time_centers<=Behavior.End(all_this_behaviors(bn_n));
                behavior_matrix(bn,pb,:) = squeeze(behavior_matrix(bn,pb,:))'+index_with_behavior;
                end
            end
        end
    all_behavior_list = [all_behavior_list;all_beh_type];

        if bf==1
            play_Structure.behavior_matrix  = behavior_matrix;
            play_Structure.behavior_list    = all_beh_type;
        else
            play_Structure(bf).behavior_matrix  = behavior_matrix;
            play_Structure(bf).behavior_list    = all_beh_type;
        end
       

end


%%

behavior_list = unique(all_behavior_list);


all_behavior_merged = cell(1,numel(behavior_list));

for bn=1:numel(all_behavior_merged)

    for bf=1:numel(play_Structure)
    this_session_np = size(play_Structure(bf).behavior_matrix,2);
      what_behavior = find(ismember(play_Structure(bf).behavior_list,behavior_list{bn}));

    if isempty(what_behavior)
        this_matrix = nan(size(play_Structure(bf).behavior_matrix,2),size(play_Structure(bf).behavior_matrix,3));
    else
        this_matrix = squeeze(play_Structure(bf).behavior_matrix(what_behavior,:,:));
    end
    all_behavior_merged{bn} = [all_behavior_merged{bn};this_matrix];
    end

end

%%
plot_bool = false;
mean_all_behaviors = [];
for  bn=1:numel(all_behavior_merged)
    this_behavior = all_behavior_merged{bn};
    this_behavior(this_behavior>1) = 1;
    if plot_bool
    figure
   
   
    colormap(1-gray)
    subplot(3,1,1:2)
    imagesc(1:(3*wraped_bins),1:size(all_behavior_merged{bn},1),this_behavior )
    axis xy
    title(behavior_list{bn})


    subplot(3,1,3)
   plot(1:(3*wraped_bins),  mean(this_behavior, 'omitmissing'))
    end
   mean_all_behaviors = [mean_all_behaviors;smooth(mean(this_behavior, 'omitmissing'),5)'];
end
%%
behavior2include = 2:17;
figure
mean2include =mean_all_behaviors (behavior2include,:);
mean_all_behaviors_norm = diag(1./max(mean2include,[],2))*mean2include;
rangeformax = 10:140;


[value, latency] = max(mean_all_behaviors_norm(:,rangeformax),[],2);

[sroted_latency, order]=sort(latency, 'ascend');

hold on
for bn=1:size(mean2include,1)


    plot(1:(3*wraped_bins),mean_all_behaviors_norm(order(bn),:) +(bn-1)*1.1, 'k')
    hold on
    plot(rangeformax(sroted_latency(bn)),mean_all_behaviors_norm(order(bn),rangeformax(sroted_latency(bn))) +(bn-1)*1.1, 'xr')



end

yticks((0:size(mean2include,1)-1)*1.1)
yticklabels(behavior_list(behavior2include(order)))
fill([50 100 100 50],[0 0 size(mean2include,1)*1.1 size(mean2include,1)*1.1] ,'r', 'FaceAlpha', .25, 'EdgeColor', 'none')
axis tight








