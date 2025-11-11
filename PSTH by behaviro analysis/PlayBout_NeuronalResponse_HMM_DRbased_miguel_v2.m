%% PARAMETERS

clear all
 n_shuff = 1000;
wrap_bins = 20;
time_jitter =1;
n_cat={'good','mua'};   % Cluster type  
% n_cat={'good'};
n_perm = 1000;
N = 5;
ca              = 1;
bins_per_warp   = 20;
wrap_time       = [-5 5];

sigma2=4;       % for shade plot smoothness
lim=20;
hist_ed = [-lim lim];
histogram_edges=hist_ed;
bin_size        = 0.1;

bins=diff(histogram_edges)/bin_size;


% bin_size = 0.5;
% bin_size = 1;


% post_onset=1; titlepostorpre='Post-Onset';% 1 if you want to evaluate the period posterior to the onset
% post_onset=0; titlepostorpre='Pre-Onset';

bins_to_eval=10; % use 10 for bin_size = 0.1; 5 for bin_size = 0.2
alpha_neuronal_res=0.05;

rat_to_plot=1;
pick_partner=2; % 2, 3 or 4

PlayBout=1; % Choose 1 if you want PlayBouts, 0 for individual behaviors
HMM=0; % Choose 1 if you want PlayBouts defined by the HMM
save_responses=1;
playbout_tittle='PlayBout';

beh_to_plot='Pounce_A';

length_duration_threshold=0.25;

% behaviors_BL={'Pounce_A','CC','Pin','Boxing','Evasion','CB','Pounce_B','Escape','CD','Rearing','Grooming','Scratch','Pounce_Ai','Pounce_Bi','Bite','Sniffing'}; %% Trying 07.03.2025
behaviors_BL={'CC','CB','CD','Escape','Pounce_A','Pounce_B','Pin','Boxing','Evasion','Rearing','Grooming','Sniffing','Scratch','Pounce_Ai','Pounce_Bi','Bite'}; %% Trying 27.08.2025
behaviors2check = behaviors_BL;

%% Pt 1 (s) : 
% B1S3_1008 = 250; 
% B2S2_1110 = 600; 
% B1D1_1013 = 700;
%%

% partner_1=250;
partner_1=inf;
% partner_1=0;

B1D1 = [1 2 3];
B1S3 = [4 5 6];
B2S2 = [7 8 9];
B3D2 = 10;


%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%CHECK THISSSSSS
B4D4 = [11 11] ; %   0 if medial 1 if lateral %% Dual 4rd Batch Probe 1
B4S2 = [12 12];                              %% Single 4rd Batch Probe 1

Npx2Probe={'Medial','Lateral'};
medial_lateral_probe=[0 0 0 0 0  0 0 0 0 0  1 0  0 0 0  0 1  ]; %   1 if medial and 0 if lateral
probes=[              1 1 1 1 1  1 1 1 1 0  3 1  1 1 1  1 3  ]; %    3 is medial and 1 is lateral     for B4S2, oposite if it is B4D4

MorL = {'_','_','_','_','_', '_','_','_','_','_', 'Med','Lat', 'mPFC','mPFC','mPFC' , 'Med' , 'Lat'};

thisrat=[ B1D1 B1S3 B2S2 B3D2 B4S2 B1D1 B4D4];

% thisrat = B4D4;

PAG_mPFC=[            1 1 1 1 1  1 1 1 1 1  1 1  2 2 2  1 1  ]; % 1 for PAG , 2 for mPFC

ELAN_segment_member=[20]; %[4 5 8 9 10 11 12]; % 1 2 3

IsItAttack=[0 0 ];
pre_beh_event=1;

% playfulBeh={'Pounce_A', 'Pin','Boxing','Pounce_B','CC','Evasion','CB','Escape'};
% playfulBeh={'Pounce_A', 'Pin','Boxing','Pounce_B','CC','Evasion'}; % van Kerkhof 2013

playfulBeh={'Pounce','Pin','Boxing','CC','CB','CD','Evasion','Escape'}; % 2025.07.10 % Playbout 

% playfulBeh={'Pounce','Pin','Boxing'}; % for Dorsal Raphe


%%
exp_info = readtable('\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\Experiments_info.csv');
cd('\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis')
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
beh_bin = 0.1; % Old as of 260225 = 0.01. 0.1 to match Beh_EmiSeq
conv_length = 1;
behavior_window =1;

hist_range = [0 20];
hist_bins = .1;
hist_edges = hist_range(1):hist_bins:hist_range(2);
hist_edges_centers = .5*(hist_edges(1:end-1)+hist_edges(2:end));


cc_matrix           = nan(numel(behavior_files),2*(lag_time/beh_bin) +1 );
inter_bout_matrix   = nan(numel(behavior_files),numel(hist_edges_centers));
bout_length_matrix  = nan(numel(behavior_files),numel(hist_edges_centers));
partner_specific_comb_bout_length    = [];
partner_specific_comb_bout_interval = [];
animal_comb = [];

play_bout_structure = [];

this_session_first_last_table = []
%%
ALL_behavior_matrices   = [];

% play_behaviors      = {'Pounce', 'CC', 'CB', 'Escape', 'Evasion', 'Pin','Boxing'};
% play_behaviors      = {'Pounce', 'CC', 'Evasion', 'Pin','Boxing'}; % van Kerkhof 2013
% play_behaviors      = {'Pounce'}; % van Kerkhof 2013

play_behaviors = playfulBeh;

beh_matrix_range    = [-20 20];
beh_matrix_bin      = 0.01;
beh_matrix_indexes  = beh_matrix_range(1):beh_matrix_bin:beh_matrix_range(2);
all_feature_matrix_zs = [];
all_feature_matrix=[];
all_feature_matrix_regression_matrix = [];
all_feature_matrix_regression_matrix2 = [];
session_id = [];
all_wrapped_psth          = [];
all_wrapped_psth_shuffled = [];

all_clusters_areas = [];
warning('off', 'all');

bf_i =1;
% bf_i =16;
% bf_i =11;

for bf =   thisrat(3:end)% [1:6 8:numel(behavior_files)]% 8%8:numel(behavior_files) % 7 is wrong
% bf = thisrat% [1:6 8:numel(behavior_files)]% 8%8:numel(behavior_files) % 7 is wrong
    disp(behavior_files(bf).name)    
    fprintf('\n%s1', behavior_files(bf).name); 
    all_feature_matrix_this_Session = [];
    all_feature_matrix_this_Session_regressor_matrix = [];
    all_feature_matrix_this_Session_regressor_matrix2 = [];

    all_feature_matrix_this_Session_zsc = [];
    session_id_this_Session = [];

    significant_res=[];
    heatmap_values_exc=[];
    heatmap_values_inh=[];
    depth_or_Chn_flipped=[];
    model_contribution=[];   

    Criterion={};
    est={};
    pval={};
    dev_test_p=[];
    est_full= [];
    pval_full= [];
    initial_cluster=[];
        
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
    partner_names(ismember(animal_types, repeated_animal)) = [];


    %% Add Transitions
    
    if PlayBout==1
            Table_Behavior=Behavior(:,[1 2 3  5 6]);
            behavior_added_state = add_behavior_states(Table_Behavior, 0, 'Transition', []);
            config.Behavior = behavior_added_state  ;
    else
         config.Behavior = Behavior  ;
    end
   
    config.repeated_animal =repeated_animal     ;
    config.animal_types =animal_types        ;
    config.play_behaviors =play_behaviors      ;
    config.beh_bin =beh_bin             ;
    config.conv_length =conv_length         ;
    config.behavior_window =behavior_window         ;


    

%% Sync with Npx timeline

    aux_mydate=behavior_files(bf).name(6:9);
    if str2double(behavior_files(bf).name(2))>3
        mydate=['2024' aux_mydate];
    else
        mydate=['2023' aux_mydate];
    end

% if 1  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Skip to LOAD data

    parent_path=['\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\' mydate]; % change according to day-session% 

    % if str2double(aux_mydate) == 1008 || str2double(aux_mydate) == 1009 || str2double(aux_mydate) == 1012 %% Put it in case of B1S3
    if ismember(bf,[4 5 6]) % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        load([parent_path '\synch_model_video2audio.mat'])
        if ismember(bf,[4 5])
            load([parent_path '\synch_model_NPX2audio.mat'])
        end
    else
        load([parent_path '\synch_model_video2NPX.mat'])            
        synch_model_video2audio=synch_model_video2NPX;
    end

    ani_ID           =   behavior_files(bf).name(1:4);
    find_this_animal =   find(ismember(exp_info.AnimalID,ani_ID) & ismember(exp_info.ExpDate,str2double(mydate)));
    
    find_this_animal =   find_this_animal(PAG_mPFC(bf_i));
    rec=num2str(exp_info.Rec);
    kilosort=exp_info.kilosort;
   
    rats{1}         =exp_info.RatRec1(find_this_animal);
    rats{2}=exp_info.Partner1(find_this_animal);
    rats{3}=exp_info.Partner2(find_this_animal);

    %% Play Bout or Single Play Behavior

    if PlayBout==1 & HMM == 0

        [play_table, play_bout_behaviors]  = play_bout(config) ;
        
        if ismember(thisrat,ELAN_segment_member) % 11 12
            aux=find(ismember(table2cell(config.Behavior(:,4)),'Partners session'));
            play_table=play_table(play_table(:,1)<table2array(config.Behavior(aux,3)),:);

            init_sess=table2cell(config.Behavior(aux,2));end_sess=table2cell(config.Behavior(aux,3));init_sess=init_sess{1};end_sess=end_sess{1};
        elseif partner_1==inf
            init_sess=0;
            aux=find(ismember(table2cell(config.Behavior(:,1)),rats{3}{1}));aux=aux(end);
            play_table=play_table(play_table(:,1)<table2array(config.Behavior(aux,3)),:); %% Select partner in B4D4, also by entering the partner in any other day or rat
            end_sess=table2array(config.Behavior(aux,3))+5;
        else
            aux=find(ismember(table2cell(config.Behavior(:,1)),rats{1}{1}));aux=aux(end);
            play_table=play_table(play_table(:,1)<table2array(config.Behavior(aux,3)),:); %% 
            end_sess=table2array(config.Behavior(aux,3))+5;
            init_sess=0;
        end

        beh_to_plot='PlayBout';

    elseif PlayBout==1 & HMM == 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        
        [play_table, play_bout_behaviors]  = play_bout(config) ;           % BOUTS
        
        if ismember(thisrat,ELAN_segment_member ) % 12
            aux=find(ismember(table2cell(config.Behavior(:,4)),'Partners session'));
            play_table=play_table(play_table(:,1)<table2array(config.Behavior(aux,3)),:);

            init_sess=table2cell(config.Behavior(aux,2));end_sess=table2cell(config.Behavior(aux,3));init_sess=init_sess{1};end_sess=end_sess{1};
        elseif partner_1==inf
            init_sess=0;
            aux=find(ismember(table2cell(config.Behavior(:,1)),rats{3}{1}));aux=aux(end);
            play_table=play_table(play_table(:,1)<table2array(config.Behavior(aux,3)),:); %% Select partner in B4D4, also by entering the partner in any other day or rat
            end_sess=table2array(config.Behavior(aux,3))+5;
        else
            aux=find(ismember(table2cell(config.Behavior(:,1)),rats{2}{1}));aux=aux(end);
            play_table=play_table(play_table(:,1)<table2array(config.Behavior(aux,3)),:); %% 
        end

        if ismember(bf,[4 5])                                              % STATES
            load([parent_path '\beg_end_times_audio_250s.mat'])
            play_table_states=beg_end_times;
        else
            load([parent_path '\beg_end_times_NPX.mat'])
            play_table_states=beg_end_times_NPX;
        end

        beh_to_plot='PlayBout';
    else

        [play_table,~]  = play_behavior(config,beh_to_plot,length_duration_threshold,rats,rat_to_plot, pick_partner,IsItAttack,pre_beh_event,playfulBeh) ;
    end

    BegTime = []; EndTime = [];
    if PlayBout==1 & HMM == 0        
        BegTime = predict(synch_model_video2audio, play_table(:,1)); %% 
        EndTime = predict(synch_model_video2audio, play_table(:,2));
        Behavior.Start = predict(synch_model_video2audio, Behavior.Start);
        Behavior.End = predict(synch_model_video2audio, Behavior.End);
    else
        disp('no synch')
        BegTime = play_table(:,1); %% 
        EndTime = play_table(:,2);
    end
    
    % Filter Beh events or states by length
    filtered_length=EndTime-BegTime;
    filtered_length=filtered_length>length_duration_threshold;

    BegTime = BegTime(filtered_length);
    EndTime = EndTime(filtered_length);
   

    %DEfine numbe rof partner and partner sessions
    partner_per_pb = nan(size(play_table,1),1);


    partner_list_names = config.animal_types;
    partner_list_names(ismember(partner_list_names, 'Session_structure')) = [];
    partner_list_names(ismember(partner_list_names, repeated_animal)) = [];

    n_partners = numel(partner_list_names);
    session_table = Behavior(ismember(Behavior.Animal , 'Session_structure') & ismember(Behavior.Type, 'Partners session'),{'Start','End'});
    session_table.Partner = nan(n_partners,1);
    percentage_per_session = nan(size(session_table,1),n_partners);
    for sn_index=1:size(session_table,1)
        for pn=1:n_partners

            percentage_per_session(sn_index,pn)=sum(ismember(Behavior.Animal(Behavior.Start>session_table.Start(sn_index) & Behavior.End<session_table.End(sn_index)),partner_list_names{pn}));
        end
    end

    [~,partner_session] = max(percentage_per_session,[],2);

    session_table.Partner  =partner_session;


   

%% Extract Clusters information
   
    
    if str2double(behavior_files(bf).name(2))>3
        path=[parent_path '\' exp_info.OpenEphys{find_this_animal} '\Record Node 101\experiment1\recording' rec(find_this_animal) '\continuous\Neuropix-PXI-100.' exp_info.ProbeNpx2{find_this_animal} '\' kilosort{find_this_animal} '\']; 
    else
        path=[parent_path '\' exp_info.OpenEphys{find_this_animal} '\Record Node 101\experiment1\recording' rec(find_this_animal) '\continuous\Neuropix-PXI-100.' num2str(exp_info.Probe(find_this_animal)) '\' kilosort{find_this_animal} '\']; 
    end

    % templates=[];
    
    spike_times     = readNPY([path 'spike_times.npy']);
    spike_clusters  = readNPY([path 'spike_clusters.npy']);
    cluster_data    = tdfread([path 'cluster_info.tsv']);
    cluster_group   = tdfread([path 'cluster_group.tsv']);
    cluster_KSLabel = tdfread([path 'cluster_KSLabel.tsv']);
    % templates = readNPY([path 'templates.npy']);   % size: [nTemplates x nTimepoints x nChannels]
    % spike_templates = readNPY('spike_templates.npy'); % spike -> template ID
    % templates_ind = readNPY('templates_ind.npy'); % channels per template

    % ch_map   = readNPY([path 'channel_map.npy']);
    % ch_pos   = readNPY([path 'channel_positions.npy']);

    good_clusters=[];
    ch_number=[];
    depth_or_Chn=[];
    cl_label=[];
    Chn=[];
    AvFR=[];
    templates_aux = [];

    for cl_cat=1:numel(n_cat)
    
        good_clusters   = [good_clusters cluster_data.cluster_id(ismember(cluster_data.group, n_cat(cl_cat)))'];
        ch_number = [ch_number cluster_data.ch(ismember(cluster_data.group, n_cat(cl_cat)))'];
    
        Chn = [Chn cluster_data.ch(ismember(cluster_data.group, n_cat(cl_cat)))'];
        depth_or_Chn = [depth_or_Chn cluster_data.depth(ismember(cluster_data.group, n_cat(cl_cat)))'];

        AvFR = [AvFR cluster_data.fr(ismember(cluster_data.group, n_cat(cl_cat)))'];
        
        sel_cl=ismember(cluster_data.group, n_cat(cl_cat));
        cl_label= [cl_label, cluster_data.group(sel_cl,:)'];

        % templates_aux = [templates_aux; templates(ismember(cluster_data.group, n_cat(cl_cat)),:,:)];

    end

    cl_label=cl_label';

%% Divide by Recorded Areas

    areas={};

    
    if strcmp(ani_ID,'B1D1')
        if bf_i > 12 & bf_i < 16
            ratID={'Batch1Dual1'};probe=probes(bf_i);probe_area='mPFC';
        else
            ratID={'Batch1Dual1'};probe=probes(bf_i);probe_area='PAG';
        end
    elseif strcmp(ani_ID,'B1S3')
        ratID={'Batch1Single3'};probe=probes(bf_i);probe_area='PAG';
    elseif strcmp(ani_ID,'B2S2')
        ratID={'Batch2Single2'};probe=probes(bf_i);probe_area='PAG';
    elseif strcmp(ani_ID,'B3D2')
        ratID={'Batch3Dual2'};probe=probes(bf_i);probe_area='PAG';
    elseif strcmp(ani_ID,'B4S2')
        if bf_i == 11
             ratID={'Batch4Single2'};probe=probes(bf_i);probe_area='PAG';
        elseif bf_i == 12
            ratID={'Batch4Single2'};probe=probes(bf_i);probe_area='PAG';
        end
    elseif strcmp(ani_ID,'B4D4')
        if bf_i == 16
             ratID={'Batch4Dual4'};probe=probes(bf_i);probe_area='PAG';
        elseif bf_i == 17
            ratID={'Batch4Dual4'};probe=probes(bf_i);probe_area='PAG';
        end
    end


       PAG_columns=[];
       start_out=[];
       [start_out,AREAS,allareas] = take_PAG_area(ratID,probe,probe_area,areas);
       for k=1:numel(start_out)
            PAG_columns(k)=start_out(k);
       end

       %% Order by Depth

        
    

    if strcmp(ani_ID(1:2),'B4') %& bf_i==11

        odd_or_even=(1:2:384)+medial_lateral_probe(bf_i);
        sel_Chn=ismember(ch_number,odd_or_even);
        good_clusters=good_clusters(sel_Chn);
        depth_or_Chn=depth_or_Chn(sel_Chn);

        [ind1, order1]=sort(depth_or_Chn);
        good_clusters=good_clusters(order1);
        depth_or_Chn=ind1;

    else
        [ind1, order1]=sort(depth_or_Chn);
        ch_number=ch_number(order1);
        good_clusters=good_clusters(order1);
        cl_label=cl_label(order1,:);
        depth_or_Chn=ind1;
        AvFR=AvFR(order1);

    end

    actual_areas=[];

        j=1;
        for i = 1:numel(PAG_columns)
            if i<numel(PAG_columns)
                aux_sel = find(isbetween_MM(depth_or_Chn,PAG_columns(i),PAG_columns(i+1)));
            else
                aux_sel = find(isbetween_MM(depth_or_Chn,PAG_columns(end),depth_or_Chn(end)));
            end

            if isempty(aux_sel)
                aux_sel=0;
            else
                initial_cluster(j)=aux_sel(end);
                actual_areas(i)=aux_sel(end);

                j=j+1;
            end
        end

        actual_areas=actual_areas>0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  INITIALIZE VARIABLES using a loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    all_psth_max_norm                       = zeros(numel(good_clusters),bins);
    all_psth_zscore                         = zeros(numel(good_clusters),bins);
    all_psth_FR                             = zeros(numel(good_clusters),bins);
    all_psth_PB_warped                      = zeros(numel(good_clusters),3*bins_per_warp);
    all_psth_PB_warped_corrected            = zeros(numel(good_clusters),3*bins_per_warp);
    all_psth_FR_pblimit                     = zeros(numel(good_clusters),bins);
    all_psth_FR_offset                      = zeros(numel(good_clusters),bins);
    all_psth_zscore_offset                  = zeros(numel(good_clusters),bins);
    all_psth_shuffled_PB_warped             = cell(numel(good_clusters),1);
    all_psth_shuffled_PB_warped_corrected   = cell(numel(good_clusters),1);
    all_mean_std                            = zeros(numel(good_clusters),3);
    
    play_behaviors_type1 = {'Pounce_A' 'Pounce_B' 'Pin'	'Boxing'	'CC'	'CB'	'CD'	'Evasion'	'Escape', 'Boxing'};
        
    behavior_types = {play_behaviors_type1, 'CC', 'Pounce_A','Pounce_B', 'Pounce_Ai', 'Pounce_Bi', 'Evasion', ...
        'Rearing','Escape', 'CD', 'Sniffing', 'CB', 'Grooming','Pin', 'Boxing', 'Scratch','Bite'};

    behavior_labels = {'play', 'CH', 'POA','POB', 'PWIA','PWIB', 'EV', ...
        'RE',  'ES', 'CD', 'SN', 'CB', 'GR', 'PI','BO', 'SC','BI'};

    
    for b = 1:numel(behavior_labels)
        label = behavior_labels{b};
        if strcmp(label, 'play')
            % Base case (no suffix)
            all_psth_self_warped  = zeros(numel(good_clusters), 3 * bins_per_warp);
            all_psth_other_warped = zeros(numel(good_clusters), 3 * bins_per_warp);
            all_psth_shuffled_self_warped  = cell(numel(good_clusters), 1);
            all_psth_shuffled_other_warped = cell(numel(good_clusters), 1);
        else
            % With suffix
            eval(sprintf('all_psth_self_warped_%s  = zeros(numel(good_clusters), 3 * bins_per_warp);', label));
            eval(sprintf('all_psth_other_warped_%s = zeros(numel(good_clusters), 3 * bins_per_warp);', label));
            eval(sprintf('all_psth_shuffled_self_warped_%s  = cell(numel(good_clusters), 1);', label));
            eval(sprintf('all_psth_shuffled_other_warped_%s = cell(numel(good_clusters), 1);', label));
        end
    end

for pn=1:2
    eval(sprintf('all_psth_self_warped_%s = zeros(numel(good_clusters), 3 * bins_per_warp);', ['Partner', num2str(pn)]));
    eval(sprintf('all_psth_other_warped_%s = zeros(numel(good_clusters), 3 * bins_per_warp);', ['Partner', num2str(pn)]));
    eval(sprintf('all_psth_PB_warped_%s= zeros(numel(good_clusters),3 * bins_per_warp);', ['Partner', num2str(pn)]));
    eval(sprintf('all_psth_shuffled_self_warped_%s = cell(numel(good_clusters), 1);', ['Partner', num2str(pn)]));
    eval(sprintf('all_psth_shuffled_other_warped_%s = cell(numel(good_clusters), 1);', ['Partner', num2str(pn)]));     
    eval(sprintf('all_psth_shuffled_PB_warped_%s= cell(numel(good_clusters), 1);', ['Partner', num2str(pn)]));
end
    % define variables to save (names)
    vars_to_save = { ...
        "initial_cluster", "AREAS", "allareas", ...
        "all_psth_zscore", "all_psth_zscore_offset", "all_psth_FR_pblimit", ...
        "all_psth_PB_warped","all_psth_PB_warped_corrected", "good_clusters", "depth_or_Chn", ...
        "all_psth_FR", "all_psth_FR_offset",...
        "all_mean_std","all_psth_shuffled_PB_warped","all_psth_shuffled_PB_warped_corrected"};

    % Add the common base variables
    vars_to_save = [vars_to_save, ...
        "all_psth_self_warped", "all_psth_other_warped","all_psth_shuffled_self_warped","all_psth_shuffled_other_warped"];

    % Add the per-behavior ones dynamically
    for b = 2:numel(behavior_labels) % start from 2 to skip 'play' (already added above)
        label = behavior_labels{b};
        vars_to_save = [vars_to_save, ...
            sprintf("all_psth_self_warped_%s", label), ...
            sprintf("all_psth_other_warped_%s", label)...
            sprintf("all_psth_shuffled_self_warped_%s", label), ...
            sprintf("all_psth_shuffled_other_warped_%s", label)];
    end

    for pn = 1:n_partners
        partner_label = sprintf('Partner%d', pn);

        vars_to_save = [vars_to_save, ...           
            sprintf('all_psth_self_warped_%s', partner_label), ...
            sprintf('all_psth_other_warped_%s', partner_label), ...
            sprintf('all_psth_shuffled_self_warped_%s', partner_label), ...
            sprintf('all_psth_shuffled_other_warped_%s', partner_label), ...                       
            sprintf('all_psth_PB_warped_%s', partner_label), ...
            sprintf('all_psth_shuffled_PB_warped_%s', partner_label) ...
            ];
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% estimate
    significant_res_dynamics=[];
    excited=[];
    inhibited=[];

        nClust = numel(good_clusters); 
        



        av_FR_nonbehavior=[];



        all_this_psth=[];
        all_this_psth_shuffled=[];
        width_ms = [];
        COV=[];
        width_ms_long = [];
        mean_wf = [];
        this_psth_shuffled = {};


        prefix = 'Processing cluster ';      % fixed text
        fprintf('\n%s1', prefix);             % print first number

    

        for cluster_n=1:numel(good_clusters)

            if cluster_n>1
             prev_num_digits = floor(log10(cluster_n-1)) + 1;

            % backspace previous digits
            for k = 1:prev_num_digits
                fprintf('\b');
            end

            % print current number
            fprintf('%d', cluster_n);
            pause(0.1);
            end


            path_raw_WF=[parent_path '\' exp_info.OpenEphys{find_this_animal} '\Record Node 101\experiment1\recording' rec(find_this_animal) '\continuous\Neuropix-PXI-100.' num2str(exp_info.Probe(find_this_animal)) '\continuous.dat'];

            if ismember(bf,[4 5]) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                spikes_times_audio = predict(synch_model_NPX2audio,double(spike_times(spike_clusters==good_clusters(cluster_n)))/30000 );
            elseif ismember(bf,[1 2 3 6 7 8 9 10  11 12  13 14 15  16 17])
                spikes_times_audio = double(spike_times(spike_clusters==good_clusters(cluster_n)))/30000 ;
            end
 
            % % % %
            spike_counts    = histcounts(spikes_times_audio, min(Behavior.Start(ismember(Behavior.Animal, 'Session_structure'))):bin_size:max(Behavior.End(ismember(Behavior.Animal, 'Session_structure'))));
            ratepersec      = movmean(spike_counts,1/bin_size)/bin_size;
            mean_rate       = mean(ratepersec,"omitnan");
            std_rate        = std(ratepersec,"omitnan");
            all_mean_std(cluster_n,:) = [good_clusters(cluster_n) mean_rate std_rate];
            
            histogram_edges=hist_ed;

            psth            = zeros(numel(BegTime),bins);
            psth_offset     = zeros(numel(EndTime),bins);
            fr_psth         = zeros(numel(BegTime),bins);
            fr_psth_only_pb = zeros(numel(BegTime),bins);
            fr_psth_offset  = zeros(numel(EndTime),bins);
            this_psth       = zeros(numel(BegTime),bins);
            mig_edges_centers = histogram_edges(1):bin_size:histogram_edges(2);
            mig_edges_centers = .5*(mig_edges_centers(2:end)+mig_edges_centers(1:end-1));
            before_edges = linspace(wrap_time(1),0,bins_per_warp+1 );


            for j=1:numel(BegTime)

                beg_time = BegTime(j);
                end_time = EndTime(j);         

                las_event_index = max(find(EndTime<beg_time));
                if isempty(las_event_index)
                      last_pb_end = -Inf;
                else
                    last_pb_end = EndTime(las_event_index);
                end                
                
                %%%% ONSET
                aligned_spikes = spikes_times_audio(spikes_times_audio>=beg_time+histogram_edges(1) & spikes_times_audio<=beg_time+histogram_edges(2))-beg_time;
                psth(j,:) = histcounts(aligned_spikes, histogram_edges(1):bin_size:histogram_edges(2));
                fr_psth(j,:)=(psth(j,:)/bin_size);

                fr_psth_only_pb(j,:) =  fr_psth(j,:);
                fr_psth_only_pb(j,mig_edges_centers>end_time-beg_time) =  NaN;
                fr_psth_only_pb(j,mig_edges_centers<last_pb_end-beg_time) = NaN;
                
                %%%% OFFSET
                aligned_spikes_end = spikes_times_audio(spikes_times_audio>=end_time+histogram_edges(1) & spikes_times_audio<=end_time+histogram_edges(2))-end_time;
                psth_offset(j,:) = histcounts(aligned_spikes_end, histogram_edges(1):bin_size:histogram_edges(2)); 
                fr_psth_offset(j,:)=(psth_offset(j,:)/bin_size);   



            end
           
            
            %%%%%%%%%%%%%%%%%%                  %%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%% PALYOBUTS WRAPED %%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%                  %%%%%%%%%%%%%%%%%%
           
            

            %% Preallocate structures for clarity
            

            %% Compute PWIs for each behavior (self & other)
            for b = 1:numel(behavior_types)
                beh = behavior_types{b};
                label = behavior_labels{b};
                          

                % --- Compute for self and other ---
                [mean_self,  mean_self_rand]  = compute_mean_and_shuffled(spikes_times_audio, Behavior, beh, repeated_animal, N, wrap_bins, time_jitter, n_shuff, mean_rate, std_rate, -Inf, Inf,length_duration_threshold);
                [mean_other, mean_other_rand] = compute_mean_and_shuffled(spikes_times_audio, Behavior, beh, partner_list_names, N, wrap_bins, time_jitter, n_shuff, mean_rate, std_rate, -Inf, Inf,length_duration_threshold);

                % --- Save results ---
                if strcmp(label, 'play')
                    all_psth_self_warped(cluster_n,:)           = mean_self;
                    all_psth_other_warped(cluster_n,:)          = mean_other;
                    all_psth_shuffled_self_warped{cluster_n}    = mean_self_rand;
                    all_psth_shuffled_other_warped{cluster_n}   = mean_other_rand;
                else
                    eval(sprintf('all_psth_self_warped_%s(cluster_n,:) = mean_self;', label));
                    eval(sprintf('all_psth_other_warped_%s(cluster_n,:) = mean_other;', label));
                    eval(sprintf('all_psth_shuffled_self_warped_%s{cluster_n} = mean_self_rand;', label));
                    eval(sprintf('all_psth_shuffled_other_warped_%s{cluster_n} = mean_other_rand;', label));
                end
            end


            %% Store means in all_psth_*_warped arrays
           
           

            % first playbouts           
            [this_psth_wraped, ~,~, ~, ~] = estimate_wrapped_psth(spikes_times_audio, BegTime, EndTime, N, N, [1 1 1]*wrap_bins, time_jitter,false,false);
            this_psth_wrped_shuffled = nan(n_shuff, size(this_psth_wraped, 2));
            for sh = 1:n_shuff
                [~, psth_shuffled,~, ~, ~] = estimate_wrapped_psth(spikes_times_audio, BegTime, EndTime, N, N, [1 1 1]*wrap_bins, time_jitter,true,false);
                psth_shuffled = (mean(psth_shuffled, "omitnan") - mean_rate) / std_rate;
                this_psth_wrped_shuffled(sh,:) = psth_shuffled;
            end



            % second  playbouts removing spikes with previous playbouts
            [this_psth_wraped_corrected, ~,~, ~, ~] = estimate_wrapped_psth(spikes_times_audio, BegTime, EndTime, N, N, [1 1 1]*wrap_bins, time_jitter,false,true);
            this_psth_wrped_shuffled_corrected = nan(n_shuff, size(this_psth_wraped_corrected, 2));
            for sh = 1:n_shuff
                [~, psth_shuffled,~, ~, ~] = estimate_wrapped_psth(spikes_times_audio, BegTime, EndTime, N, N, [1 1 1]*wrap_bins, time_jitter,true,true);
                psth_shuffled = (mean(psth_shuffled, "omitnan") - mean_rate) / std_rate;
                this_psth_wrped_shuffled(sh,:) = psth_shuffled;
            end
  
           

            for pn = 1:n_partners

                partner_number = session_table.Partner(pn);
                T1 = session_table.Start(pn);
                T2 = session_table.End(pn);



                % --- Compute for self and other ---
                [mean_self,  mean_self_rand]  = compute_mean_and_shuffled(spikes_times_audio, Behavior, play_behaviors, repeated_animal, N, wrap_bins, time_jitter, n_shuff, mean_rate, std_rate, T1, T2,length_duration_threshold);
                [mean_other, mean_other_rand] = compute_mean_and_shuffled(spikes_times_audio, Behavior, beh, partner_list_names(partner_number), N, wrap_bins, time_jitter, n_shuff, mean_rate, std_rate, T1, T2,length_duration_threshold);

                % --- Save results ---

                eval(sprintf('all_psth_self_warped_%s(cluster_n,:) = mean_self;', ['Partner', num2str(pn)]));
                eval(sprintf('all_psth_other_warped_%s(cluster_n,:) = mean_other;', ['Partner', num2str(pn)]));               
                eval(sprintf('all_psth_shuffled_self_warped_%s{cluster_n} = mean_self_rand;', ['Partner', num2str(pn)]));
                eval(sprintf('all_psth_shuffled_other_warped_%s{cluster_n} = mean_other_rand;', ['Partner', num2str(pn)]));


                index = BegTime>=T1 & EndTime<=T2;

                [this_partner_psth_wrped, ~,~, ~, ~] = estimate_wrapped_psth(spikes_times_audio, BegTime(index), EndTime(index), N, N, [1 1 1]*wrap_bins, time_jitter,false,false);
                mean_this_partner = (mean(this_partner_psth_wrped,"omitnan")-mean_rate)/std_rate;
                this_partner_wrped_shuffled = nan(n_shuff, size(this_partner_psth_wrped, 2));
                for sh = 1:n_shuff
                    [~, psth_shuffled,~, ~, ~] = estimate_wrapped_psth(spikes_times_audio, BegTime(index), EndTime(index), N, N, [1 1 1]*wrap_bins, time_jitter,true,false);
                    psth_shuffled = (mean(psth_shuffled, "omitnan") - mean_rate) / std_rate;
                    this_partner_wrped_shuffled(sh,:) = psth_shuffled;
                end


                eval(sprintf('all_psth_PB_warped_%s(cluster_n,:)        = mean_this_partner;', ['Partner', num2str(pn)]));
                eval(sprintf('all_psth_shuffled_PB_warped_%s{cluster_n} =this_partner_wrped_shuffled;', ['Partner', num2str(pn)]));



            end



             
            all_psth_PB_warped(cluster_n,:)                     = (mean(this_psth_wraped,"omitnan") - mean_rate) / std_rate ;  
            all_psth_PB_warped_corrected(cluster_n,:)           = (mean(this_psth_wraped_corrected,"omitnan") - mean_rate) / std_rate ; 
            all_psth_shuffled_PB_warped{cluster_n}              = this_psth_wrped_shuffled;
            all_psth_shuffled_PB_warped_corrected{cluster_n}    = this_psth_wrped_shuffled_corrected  ;
            all_psth_FR(cluster_n,:)                            = mean(fr_psth,"omitnan");
            all_psth_FR_offset(cluster_n,:)                     = mean(fr_psth_offset,"omitnan");
            all_psth_FR_pblimit(cluster_n,:)                    = mean(fr_psth_only_pb,"omitnan");
            all_psth_zscore(cluster_n,:)                        = zscore(mean(psth,"omitnan"));
            all_psth_zscore_offset(cluster_n,:)                 = zscore(mean(psth_offset,"omitnan"));

            

        RM_areas=AREAS(actual_areas);

       
        end
        all_mean_std = array2table(all_mean_std);
        all_mean_std.Properties .VariableNames = {'ClusterID','MeanRate','STD'};
         if save_responses==1

            if  HMM==0

                % path='\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\Responses_Matrix\ModelCriterion\';
                path='\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\Responses_Matrix\ModelCriterion_Onset_Miguel\';

                save([path 'ResponsesMatrix_PPB_p1andp2_' num2str(length_duration_threshold) 's_' playbout_tittle '_' mydate '_' behavior_files(bf).name(1:4) '_' MorL{bf_i} '.mat'],...
                   vars_to_save{:})

                % save([path 'ResponsesMatrix_PPB_p1andp2_' num2str(length_duration_threshold) 's_Gcl' playbout_tittle '_' mydate '_' behavior_files(bf).name(1:4) '_' MorL{bf_i} '.mat'],"initial_cluster","AREAS","RM_areas","all_psth_zscore", ...
                %     "good_clusters_flipped","depth_or_Chn_flipped","pre_onset","post_onset","pre_offset","post_offset","all_psth_zscore_offset","all_features","all_this_psth","all_this_psth_shuffled","est_full_PPB","pval_full_PPB","mod_wrap", ...
                %     "good_clusters","depth_or_Chn","all_psth_FR","all_psth_FR_offset","AvFR","av_FR_nonbehavior","width_ms","COV","width_ms_long","mean_wf","wfRaw")

            else
                % path='\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\Responses_Matrix\ModelCriterion\';
                path='\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\Responses_Matrix\ModelCriterion_Onset_Miguel\';


                save([path 'ResponsesMatrix_PPB_HMM_p1_' num2str(length_duration_threshold) 's_' playbout_tittle '_' mydate '_' behavior_files(bf).name(1:4) '_' MorL{bf_i}  '.mat'],...
                     vars_to_save{:})

            end
        end

        bf_i = bf_i + 1;


end


% function PWI = compute_PWI(Behavior, spikes_times_audio, animal_list, bins_per_warp, wrap_time, before_edges, behavior_name)
%     % Helper to compute peri-event spike rate (PWI) around 'PounceI' events
%     %
%     % Inputs:
%     %   Behavior           - table with fields: Type2, Animal, Start, End
%     %   spikes_times_audio - vector of spike timestamps
%     %   animal_list        - list of animal names to include
%     %   bins_per_warp      - number of bins per warp window
%     %   wrap_time          - [before after] time window around event
%     %   before_edges       - bin edges for pre-event window (typically linspace(wrap_time(1),0,bins_per_warp+1))
%     %
%     % Output:
%     %   PWI - nEvents x (3*bins_per_warp) matrix of normalized firing rates
% 
%     % find matching events
%     play_inds = find(ismember(Behavior.Type,behavior_name ) & ismember(Behavior.Animal, animal_list));
% 
%     % preallocate
%     PWI = nan(numel(play_inds), 3 * bins_per_warp);
% 
%     % loop over events
%     for j = 1:numel(play_inds)
%         beg_time = Behavior.Start(play_inds(j));
%         end_time = Behavior.End(play_inds(j));
% 
%         during_edges = linspace(0, end_time - beg_time, bins_per_warp + 1);
%         after_edges  = linspace(end_time - beg_time, end_time - beg_time + wrap_time(2), bins_per_warp + 1);
% 
%         aligned_spikes_before = spikes_times_audio(spikes_times_audio >= beg_time + wrap_time(1) & spikes_times_audio <= beg_time) - beg_time;
%         counts_before = bins_per_warp * histcounts(aligned_spikes_before, before_edges) / abs(wrap_time(1));
% 
%         aligned_spikes_during = spikes_times_audio(spikes_times_audio >= beg_time & spikes_times_audio <= end_time) - beg_time;
%         counts_during = bins_per_warp * histcounts(aligned_spikes_during, during_edges) / (end_time - beg_time);
% 
%         aligned_spikes_after = spikes_times_audio(spikes_times_audio >= end_time & spikes_times_audio <= end_time + wrap_time(2)) - beg_time;
%         counts_after = bins_per_warp * histcounts(aligned_spikes_after, after_edges) / abs(wrap_time(2));
% 
%         PWI(j,:) = [counts_before counts_during counts_after];
%     end
% 
% 
% 
% end


function [psth_wrapped,psth_shuffled, time_wrapped, rate_during , psth_counts_real ] = estimate_wrapped_psth(spike_times, event_times_onset, event_times_offset, t_before, t_after, wrap_bins,time_jitter, shuffled, clear_onset_offset)
% Inputs:
%   - spike_times: vector of spike timestamps
%   - event_times_onset: vector of onset times (1 per event)
%   - event_times_offset: vector of offset times (same size)
%   - t_before: time before onset to include
%   - t_after: time after offset to include
%   - wrap_bins: [n1 n2 n3] number of bins for each segment: pre, during, post

n_events = numel(event_times_onset);
n1 = wrap_bins(1); n2 = wrap_bins(2); n3 = wrap_bins(3);
total_bins = n1 + n2 + n3;

psth_wrapped = zeros(n_events, total_bins);
psth_counts_real = zeros(n_events, total_bins);

psth_shuffled = zeros(n_events, total_bins);

rate_during = nan(n_events,1);

bin_length_p1 = t_before/n1;
bin_length_p3 = t_after/n3;
mean_outside_length = .5*(bin_length_p1+bin_length_p3);
if clear_onset_offset==false
    for i = 1:n_events
        % Define time boundaries
        t1 = event_times_onset(i) - t_before;
        t2 = event_times_onset(i);
        t3 = event_times_offset(i);
        t4 = event_times_offset(i) + t_after;
        bin_length_p2 = (t3-t2)/n2;
        % Define bin edges for each section
        edges1 = linspace(t1, t2, n1 + 1);  % t1 to t2
        edges2 = linspace(t2, t3, n2 + 1);  % t2 to t3
        edges3 = linspace(t3, t4, n3 + 1);  % t3 to t4

        all_edges = [edges1(1:end-1), edges2(1:end-1), edges3(1:end-1), edges3(end)];

        % Select spikes in the peri-event window
        spikes_in_window = spike_times(spike_times >= t1 & spike_times < t4);
        rate_during(i) = sum((spike_times >= t2 & spike_times <= t3))/(t3 - t2);

        % Jitter each spike randomly within the peri-event window
        % jittered_spikes = t1 + rand(size(spikes_in_window)) * (t4 - t1);

        if ~shuffled
            % Bin actual  spike counts
            counts_real                     = histcounts(spikes_in_window, all_edges);
            psth_counts_real(i,:)           = counts_real;
            psth_wrapped(i, 1:n1)           = counts_real(1:n1)/bin_length_p1;
            psth_wrapped(i, (n1+1):(n1+n2)) = movmean(counts_real((n1+1):(n1+n2))/bin_length_p2, ceil(mean_outside_length/bin_length_p2));
            psth_wrapped(i, (n1+n2+1):end)  = counts_real( (n1+n2+1):end)/bin_length_p3;
        else
            if isinf(time_jitter)
                this_jitter = t4-t1;
            else
                this_jitter =time_jitter;
            end
            % Bin jittered  spike counts
            jittered_spikes = mod(spikes_in_window - t1 + rand(size(spikes_in_window))*this_jitter,(t4 - t1) ) + t1;
            counts_shuf = histcounts(jittered_spikes, all_edges);
            psth_shuffled(i, 1:n1)           = counts_shuf(1:n1)/bin_length_p1;
            psth_shuffled(i, (n1+1):(n1+n2)) = movmean(counts_shuf((n1+1):(n1+n2))/bin_length_p2, ceil(mean_outside_length/bin_length_p2));
            psth_shuffled(i, (n1+n2+1):end)  = counts_shuf((n1+n2+1):end)/bin_length_p3;
        end
    end
else
    for i = 1:n_events

        % Define time boundaries     
        t1 = event_times_onset(i) - t_before;
        t2 = event_times_onset(i);
        t3 = event_times_offset(i);
        t4 = event_times_offset(i) + t_after;
        bin_length_p2 = (t3-t2)/n2;

        %Define previous and next event
           previous_event = max(event_times_offset(event_times_offset<event_times_onset(i)));
            next_event = min(event_times_onset(event_times_onset>event_times_offset(i)));
            if isempty(previous_event)
                previous_event = -Inf;
            end
             if isempty(next_event)
                next_event = Inf;
            end


        % Define bin edges for each section
        edges1 = linspace(t1, t2, n1 + 1);  % t1 to t2
        edges2 = linspace(t2, t3, n2 + 1);  % t2 to t3
        edges3 = linspace(t3, t4, n3 + 1);  % t3 to t4

        all_edges = [edges1(1:end-1), edges2(1:end-1), edges3(1:end-1), edges3(end)];
        all_edges_centers = .5*(all_edges(2:end) + all_edges(1:end-1));

        % Select spikes in the peri-event window
        spikes_in_window = spike_times(spike_times >= t1 & spike_times < t4);
        rate_during(i) = sum((spike_times >= t2 & spike_times <= t3))/(t3 - t2);

        % Jitter each spike randomly within the peri-event window
        % jittered_spikes = t1 + rand(size(spikes_in_window)) * (t4 - t1);

        if ~shuffled
            % Bin actual  spike counts
            counts_real                     = histcounts(spikes_in_window, all_edges);
            psth_counts_real(i,:)           = counts_real;

            psth_wrapped(i, 1:n1)           = counts_real(1:n1)/bin_length_p1;
            psth_wrapped(i, (n1+1):(n1+n2)) = movmean(counts_real((n1+1):(n1+n2))/bin_length_p2, ceil(mean_outside_length/bin_length_p2));
            psth_wrapped(i, (n1+n2+1):end)  = counts_real((n1+n2+1):end)/bin_length_p3;

            psth_wrapped(i,all_edges_centers<=previous_event) = NaN; 
            psth_wrapped(i,all_edges_centers>=next_event) = NaN;

        else
            if isinf(time_jitter)
                this_jitter = t4-t1;
            else
                this_jitter =time_jitter;
            end
            % Bin jittered  spike counts
            jittered_spikes = mod(spikes_in_window - t1 + rand(size(spikes_in_window))*this_jitter,(t4 - t1) ) + t1;
            counts_shuf = histcounts(jittered_spikes, all_edges);
            psth_shuffled(i, 1:n1)           = counts_shuf(1:n1)/bin_length_p1;
            psth_shuffled(i, (n1+1):(n1+n2)) = movmean(counts_shuf((n1+1):(n1+n2))/bin_length_p2, ceil(mean_outside_length/bin_length_p2));
            psth_shuffled(i, (n1+n2+1):end)  = counts_shuf((n1+n2+1):end)/bin_length_p3;

            psth_shuffled(i,all_edges_centers<=previous_event) = NaN; 
            psth_shuffled(i,all_edges_centers>=next_event) = NaN;
        end
    end
end
% rate_during = mean(rate_during);
% Create normalized time vector: [0 → 1] for t1-t2, [1 → 2] for t2-t3, [2 → 3] for t3-t4
% Time ranges
t1 = 0;    % corresponds to event_onset - t_before
t2 = 1;    % corresponds to event_onset
t3 = 2;    % corresponds to event_offset
t4 = 3;    % corresponds to event_offset + t_after

% Number of bins per segment


% Bin edges for each segment in wrapped time
edges1 = linspace(t1, t2, n1 + 1);
edges2 = linspace(t2, t3, n2 + 1);
edges3 = linspace(t3, t4, n3 + 1);

% Compute centers
centers1 = (edges1(1:end-1) + edges1(2:end)) / 2;
centers2 = (edges2(1:end-1) + edges2(2:end)) / 2;
centers3 = (edges3(1:end-1) + edges3(2:end)) / 2;

% Concatenate
time_wrapped = [centers1, centers2, centers3];

end


function [mean_psth, mean_rand] = compute_mean_and_shuffled( ...
    spikes_times_audio, Behavior, beh, animal_list, ...
    N, wrap_bins, time_jitter, n_shuff, mean_rate, std_rate, T1, T2,length_duration_threshold)

% COMPUTE_MEAN_AND_SHUFFLED
% -------------------------------------------------------------
% Computes the mean PSTH (wrapped) and a shuffled distribution
% for a given animal or group of animals performing a behavior.
%
% Inputs:
%   spikes_times_audio : vector of spike times
%   Behavior            : table with .Start, .End, .Type2, .Animal
%   beh                 : string, behavior type to extract
%   animal_list         : string or cell array of animal IDs to include
%   N                   : window length before/after event
%   wrap_bins           : number of bins per segment (pre, during, post)
%   time_jitter         : jitter window for spike shuffling
%   n_shuff             : number of shuffles
%   mean_rate, std_rate : baseline firing stats for z-scoring
%
% Outputs:
%   mean_psth : mean z-scored PSTH for the real data
%   mean_rand : [n_shuff x nBins] shuffled PSTHs (z-scored)
%
% -------------------------------------------------------------

% Extract onset and offset times for this behavior and animal(s) and filter
% by length
onset_times     = Behavior.Start(ismember(Behavior.Type, beh) & ismember(Behavior.Animal, animal_list) & Behavior.Start>=T1 & Behavior.End<T2);
offset_times    = Behavior.End(ismember(Behavior.Type, beh) & ismember(Behavior.Animal, animal_list)  & Behavior.Start>=T1 & Behavior.End<T2);
lengths         = offset_times-onset_times;
onset_times     = onset_times(lengths>=length_duration_threshold);
offset_times    = offset_times(lengths>=length_duration_threshold);

% Handle cases with no events
if isempty(onset_times)
    mean_psth = nan(1, 3 * wrap_bins);
    mean_rand = nan(n_shuff, 3 * wrap_bins);
    return;
end

% Compute wrapped PSTH for real data
[psth, ~,~, ~, ~] = estimate_wrapped_psth(spikes_times_audio, onset_times, offset_times, N, N, [1 1 1]*wrap_bins, time_jitter,false,false);
mean_psth = (mean(psth, "omitnan") - mean_rate) / std_rate;

% Compute shuffled PSTHs
mean_rand = nan(n_shuff, size(mean_psth, 2));
for sh = 1:n_shuff
    [~, psth_shuffled,~, ~, ~] = estimate_wrapped_psth(spikes_times_audio, onset_times, offset_times, N, N, [1 1 1]*wrap_bins, time_jitter,true,true);
    psth_shuffled = (mean(psth_shuffled, "omitnan") - mean_rate) / std_rate;
    mean_rand(sh,:) = psth_shuffled;
end
end