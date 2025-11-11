function mi_structure = GENERATE_STRUCUTRE_animal_synch_only_behavior(directory)
%% set parameters

%time range for the delta psth
time_range = [-5 10];
%lfp sampling rate
sr = 2500;
%params mutual information
n_bins_local        = 10;
n_bins_global       = 50;
t_size              = 200;
nIter               = 10000;  % number of randomizations
%params playbout
play_behaviors      = {'Pounce', 'CC','Boxing', 'Evasion','Pin', 'Escape', 'CB', 'CD'};
bin_size            = 0.01;
conv_length         = 1;

%params spectrogram
wind_length         = 1;
wind_overlap        = .990;
f                   = .1:.1:6;
freq_pow_range      = [1 4];
% wind_length         = .250;
% wind_overlap        = .240;
% f                   = 5:.1:14;
% freq_pow_range      = [6 12];
%% load data: animal info 

animal_info =directory;
animal_info = strsplit(animal_info, '\');
animal_info = animal_info{end};
animal_info = strsplit(animal_info, ' ');



%% load  data: Synch model
file_loc    = fullfile(directory,'synch_model_video2NPX.mat');
load(file_loc, 'synch_model_video2NPX')

%% load  data: Behavior
Behavior_file = dir([directory, '/ELAN*']);
Behavior_file =Behavior_file.name;
file_loc    = fullfile(directory,Behavior_file);

animal_1                            = animal_info{1};
Behavior                            = readtable(file_loc);
Behavior(:,2)                       = [];
Behavior.Properties.VariableNames   = {'Animal', 'Start', 'End', 'Length', 'Type'};



Behavior.Type2          = Behavior.Type;
Behavior.Type2(ismember(Behavior.Type2, {'Pounce_A', 'Pounce_B'}))      = {'Pounce'}; %% Merging behaviors to Type2
Behavior.Type2(ismember(Behavior.Type2, {'Pounce_Ai', 'Pounce_Bi'}))    = {'PounceI'};
Behavior.Type2(ismember( Behavior.Type2,''))                            = {'Other'};
Behavior(ismember(Behavior.Animal, 'Reversal'),:)                       = [];

animal_types            = unique(Behavior.Animal);

animal_types(ismember(animal_types,'Session_structure'))                =[];

Behavior.Start          = predict(synch_model_video2NPX, Behavior.Start);
Behavior.End            = predict(synch_model_video2NPX, Behavior.End);
mi_structure.Behavior   = Behavior;
end