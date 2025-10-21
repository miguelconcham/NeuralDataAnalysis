hmm_data_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\HMM data\HMM raw data'
labeled_data_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\HMM data\locomotive behaviors';
segmented_data_folder =  '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\HMM data\resampled_locomotive_Data';
dir_list = dir(hmm_data_folder);
dir_list(1:2) = [];
dir_list(end) = [];
spatial_property_names      = {  'Speed','AngleSpeed','AngleAcc','Acc','Wall2CenterPos'...
    'RelativeDistance','RelativeSpeed','RelativeAngleSpeed','RelativeAngleAcc', 'RelativeAcc'};
call_prop_list = {'PrincipalFrequencykHz', 'SlopekHzs', 'Sinuosity', 'DeltaFreqkHz', 'FrequencyStandardDeviationkHz'};

all_var = [spatial_property_names,'NumCalls',call_prop_list]

behaviors2check = {'Pounce_A','Pounce_Ai','Pounce_B','Pounce_Bi'};


%%
all_behavior = [];
for j=1:numel(dir_list)
    current_dir = [hmm_data_folder,'\',dir_list(j).name ];
    behavior_list = get_behavior_properties(current_dir, behaviors2check,call_prop_list,0, [-1 1]);   
    cd(hmm_data_folder)
    all_behavior = [all_behavior;behavior_list];
end
%%
var_lenght = cell2mat(cellfun(@(x) size(x,1),all_behavior(:,1), 'UniformOutput',false));
longest     = max(var_lenght);

%%
bin_size = 0.01;
figure
var2plot ={'RelativeSpeed', 'RelativeDistance','NumCalls'};
smooth_var = [0 0 1];
smoth_size = 25;
for vn = 1:numel(var2plot)

    subplot(3,numel(var2plot), (1:numel(var2plot):(2*numel(var2plot) -1)) + vn-1)


all_variables = nan(size(all_behavior,1), longest);


for j=1:size(all_behavior,1)
    if smooth_var(vn)==1
    all_variables(j,1:var_lenght(j)) = movmean(all_behavior{j,1}(:, ismember(all_var,var2plot(vn) )),smoth_size);
    else
    all_variables(j,1:var_lenght(j)) = all_behavior{j,1}(:, ismember(all_var,var2plot(vn) ));
    end
end



[ordered_length, length_order] = sort(var_lenght);


imagesc((1:longest)*bin_size  -1,1:size(all_behavior,1), all_variables(length_order,:))
axis xy
hold on
plot((ordered_length-2/bin_size)*bin_size,1:size(all_behavior,1), 'w')
hold on
plot((1:size(all_behavior,1))*0,1:size(all_behavior,1), 'w')


 subplot(3,numel(var2plot), (2*numel(var2plot) +1) + vn-1)

 plot((1:longest)*bin_size  -1, mean(all_variables, 'omitmissing'))
 xlim([-1 (longest*bin_size -2)])
end

%% call effect by length

call_ri = nan(size(all_variables,1),3);

for j=1:size(all_variables,1)
    base   = mean(all_variables(j,1:round(1/bin_size)));
    onset  = mean(all_variables(j,(round(1/bin_size)+1):round((var_lenght(j) -1/bin_size))));
    offset =  mean(all_variables(j,round((var_lenght(j)-1/bin_size)):var_lenght(j)));
    call_ri(j,:) =[base,onset,offset];
end


figure
subplot(1,2,1)
min_diff = min(call_ri(:,2)-call_ri(:,1));
[~,legth_out] = rmoutliers(var_lenght);
[~,diff_out] = rmoutliers(log(call_ri(:,2)-call_ri(:,1)-min_diff + 0.01));
data2corr = ~legth_out & ~diff_out;
semilogy(var_lenght(data2corr)*bin_size, call_ri(data2corr,2)-call_ri(data2corr,1)-min_diff + 0.01, '.')


[c,p] = corr(var_lenght(data2corr)*bin_size,log(call_ri(data2corr,2)-call_ri(data2corr,1)-min_diff + 0.01), 'Type','Spearman');



title([num2str(c), ' ', num2str(p)])

subplot(1,2,2)
min_diff = min(call_ri(:,3)-call_ri(:,1));
[~,legth_out] = rmoutliers(var_lenght);
[~,diff_out] = rmoutliers(log(call_ri(:,2)-call_ri(:,1)-min_diff + 0.01));
data2corr = ~legth_out & ~diff_out;
semilogy(var_lenght(data2corr)*bin_size, call_ri(data2corr,3)-call_ri(data2corr,1)-min_diff + 0.01, '.')


[c,p] = corr(var_lenght(data2corr)*bin_size,log(call_ri(data2corr,2)-call_ri(data2corr,1)-min_diff + 0.01), 'Type','Spearman');
title([num2str(c), ' ', num2str(p)])






