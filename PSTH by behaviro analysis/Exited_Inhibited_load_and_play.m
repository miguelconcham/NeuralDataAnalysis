n_wapred_bins = 60;
warped_time     = (((1:n_wapred_bins)/20)*5) - 5 ;
baseline = [-5 0];
baseline_index      = warped_time>=baseline(1) & warped_time<=baseline(2);

response_time = [0 5];
time_indxs      = warped_time>=response_time(1) & warped_time<=response_time(2);
all_matched_tables = [];

figure_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Codes\Figure codes\Figure 7 Inputs';
%% create psth list and theta phase data



    behavior_labels = {'play', 'CH', 'POA','POB', 'PWIA','PWIB', 'EV', ...
        'RE',  'ES', 'CD', 'SN', 'CB', 'GR'};
 
base_conditions = {'','play', 'CH', 'POA','POB', 'PWIA','PWIB', 'EV', ...
        'RE',  'ES', 'CD', 'SN', 'CB', 'GR'};
roles = {'Self', 'Other'};

psth_map = {};

% Combine your behavior and partner lists
n_partners = 2;
all_labels = [base_conditions, arrayfun(@(pn) sprintf('Partner%d', pn), 1:n_partners, 'UniformOutput', false)];

for b = 1:numel(all_labels)
    base = all_labels{b};

    % --- Handle the 'play' (empty) condition ---
    if isempty(base) || strcmp(base, 'play')
        var_self   = 'all_psth_self_warped';
        var_other  = 'all_psth_other_warped';
        field_self  = 'PsthWarpedSelf';
        field_other = 'PsthWarpedOther';
    else
        % --- Handle behavior or partner-specific cases ---
        var_self   = sprintf('all_psth_self_warped_%s', base);
        var_other  = sprintf('all_psth_other_warped_%s', base);
        field_self  = sprintf('PsthWarped%sSelf', base);
        field_other = sprintf('PsthWarped%sOther', base);
    end

    % Add mappings to the cell array
    psth_map = [psth_map; {field_self, var_self}; {field_other, var_other}];
end

%% load table directly if needed
n_wapred_bins = 60;
warped_time     = (((1:n_wapred_bins)/20)*5) - 5 ;
baseline = [-5 0];
baseline_index      = warped_time>=baseline(1) & warped_time<=baseline(2);

response_time = [0 5];
time_indxs      = warped_time>=response_time(1) & warped_time<=response_time(2);
saving_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\Theta psth';
load([saving_folder,'\all_neurons_TD.mat'],'all_neurons_TD');

%% exitedn inhibited analysis
% play_song([],[],[])
play_behaviors      = {'CH','POA','POB','EV','ES','CB','CD'};
play_behaviors      = {'CH','POA','POB','EV','ES','CB','CD'};
% play_behaviors      = {,'POA','POB'};
% play_behaviors      = {'POA'};
non_play_behaviors  = {'PWIA','PWIB','GR','SN','RE'};
% non_play_behaviors  = {'PWIA'};
% slef_other_condition = {'self','other'}
slef_other_condition = {'self'};
% non_play_behaviors  = {'PWIA','PWIB'};
 % non_play_behaviors  = {'GR','SN','RE'};
 
alpha_level_bf = 0.05/numel(play_behaviors);

exited_play_list = cell2mat(cellfun(@(x) contains(x,'Exited')  &  contains(x,play_behaviors)  &  contains(x,slef_other_condition), all_neurons_TD.Properties.VariableNames, 'UniformOutput',false));
inhibited_play_list = cell2mat(cellfun(@(x) contains(x, 'Inhibited') &  contains(x,play_behaviors) &  contains(x,slef_other_condition), all_neurons_TD.Properties.VariableNames, 'UniformOutput',false));
exited_non_play_list = cell2mat(cellfun(@(x) contains(x,'Exited')  &  contains(x,non_play_behaviors) &  contains(x,slef_other_condition), all_neurons_TD.Properties.VariableNames, 'UniformOutput',false));
inhibited_non_play_list = cell2mat(cellfun(@(x) contains(x, 'Inhibited') &  contains(x,non_play_behaviors) &  contains(x,slef_other_condition), all_neurons_TD.Properties.VariableNames, 'UniformOutput',false));

exited_play_labels = all_neurons_TD.Properties.VariableNames(exited_play_list);
inhibited_play_labels = all_neurons_TD.Properties.VariableNames(inhibited_play_list);
exited_non_play_labels = all_neurons_TD.Properties.VariableNames(exited_non_play_list);
inhibited_non_play_labels = all_neurons_TD.Properties.VariableNames(inhibited_non_play_list);

contradictory_indexes = all_neurons_TD{: ,exited_play_list}<alpha_level_bf & all_neurons_TD{: ,inhibited_play_list}<alpha_level_bf;
contradictory_indexes(isnan(contradictory_indexes)) = 1;
contradictory_indexes = contradictory_indexes==1;
exited_play         = double(all_neurons_TD{: ,exited_play_list}<alpha_level_bf);
exited_play(contradictory_indexes) = NaN;
exited_play         = sum(exited_play,2, 'omitmissing')/size(exited_play,2);


inhibited_play         = double(all_neurons_TD{: ,inhibited_play_list}<alpha_level_bf);
inhibited_play(contradictory_indexes) = NaN;
inhibited_play         = sum(inhibited_play,2, 'omitmissing')/size(inhibited_play,2);
modulated_play    = sum(exited_play+inhibited_play,2, 'omitmissing')/size(exited_play,2);

contradictory_indexes = all_neurons_TD{: ,exited_non_play_list}<alpha_level_bf & all_neurons_TD{: ,inhibited_non_play_list}<alpha_level_bf;
contradictory_indexes(isnan(contradictory_indexes)) = 1;
contradictory_indexes = contradictory_indexes==1;

exited_non_play         = double(all_neurons_TD{: ,exited_non_play_list}<alpha_level_bf);
exited_non_play(contradictory_indexes) = NaN;
exited_non_play         = sum(exited_non_play,2, 'omitmissing')/size(exited_non_play,2);


inhibited_non_play         = double(all_neurons_TD{: ,inhibited_non_play_list}<alpha_level_bf);
inhibited_non_play(contradictory_indexes) = NaN;
inhibited_non_play         = sum(inhibited_non_play,2, 'omitmissing')/size(inhibited_non_play,2);
modulated_non_play =  sum(inhibited_non_play+exited_non_play,2, 'omitmissing')/size(inhibited_non_play,2);


figure
alpha_level  = 0.01;
non_entrained_lvl = 0.1;





area_list = {'SupCol','DLPAG','LPAG','VLPAG','DR'};
cell_type = {'peak','trough','non-entrained'};

pctg_ones_per_area = nan(numel(area_list),3);
for an=1:numel(area_list)
    
    area_index              = ismember(all_neurons_TD.area, area_list{an});
    entrainment_index       = all_neurons_TD.DeltaEntireSession.PPCPval<alpha_level;
    non_entrained_index     = all_neurons_TD.DeltaEntireSession.PPCPval>non_entrained_lvl;
    peak_index              = all_neurons_TD.DeltaEntireSession.PreferedAngle>-pi/2 & all_neurons_TD.DeltaEntireSession.PreferedAngle<pi/2;
    analysed                    = ~isnan(all_neurons_TD.Exited);
    

    peak_cells          = analysed  & area_index & entrainment_index & peak_index;
    trough_cells        = analysed  & area_index & entrainment_index & ~peak_index;
    non_entrained_cells = analysed  & area_index & non_entrained_index;
    x_label = {'% of Exited', 'Playful behaviors'};
     y_label = {'% of Exited', 'Non-playful behaviors'};

    types_cell = {peak_cells,trough_cells,non_entrained_cells};
    p_per_type = nan(3,1);
    for tn= 1:3
        subplot(numel(area_list), 1, an)
        hold on
        % play_counts = [;inhibited_play(types_cell{tn})];
        % non_play_counts = [;inhibited_non_play(types_cell{tn})];
        % x12compare  = exited_play(types_cell{tn});
        % x22compare  = exited_non_play(types_cell{tn});
    % x12compare  = modulated_play(types_cell{tn} & modulated_play>0);
    %     x22compare  = modulated_play_non_play(types_cell{tn} & modulated_play>0);

        x12compare  = exited_play(types_cell{tn} & exited_play>0);
        x22compare  = exited_non_play(types_cell{tn} & exited_play>0);
        rand_x      = (rand(numel(exited_play(types_cell{tn} & exited_play>0)),2)-.5)/2;
        rand_x1      = (rand(numel(x12compare),1)-.5);
        rand_x2      = (rand(numel(x22compare),1)-.5)/5;
        what2plot =  -(x12compare-x22compare)./(x12compare+x22compare);
        
        pctg_ones_per_area(an,tn) = sum(what2plot==-1)/numel(what2plot);
        % what2plot = what2plot(what2plot~=-1)
        % what2plot = x12compare;

        % 
        % plot((rand_x + ones(numel(x12compare),2)*diag([1 2]))', [x12compare x22compare]', ':k')
        % hold on
        %  plot((rand_x + ones(numel(x22compare ),2)*diag([1 2]))', [x12compare x22compare]', '.k', 'MarkerSize',5)

       swarmchart(what2plot*0 +tn,what2plot, 'ko')

        hold on
       plot(tn, mean( what2plot, 'omitmissing'), '_r', 'MarkerSize', 10, 'LineWidth',2)
        % plot(median(x12compare), median(x22compare), 'xr', 'MarkerSize',10)
        if ~isempty(what2plot)
            [p,h] = signrank(what2plot);
        else
            p = NaN;
        end
            p_per_type(tn) = p;
        % swarmchart([play_counts*0;non_play_counts*0+1],[play_counts;non_play_counts],'.k')
        % ylim([-1 1])
        % if p<0.01
        % title(num2str(p), 'Color','r')
        % else
        %     title(num2str(p))
        % end
        % min_min = min([x22compare+rand_x1; x22compare+rand_x2]);
        % max_max = max([x22compare+rand_x1; x22compare+rand_x2]);
        % hold on
        % plot([min_min max_max],[min_min max_max], 'r')
        % axis tight
        % if tn==1
            ylabel(y_label)
        % end
        if an==numel(area_list)
            xlabel(x_label)
            xticks(1:3)
            xticklabels(cell_type)
        end
        xlim([0 4])
    end
    title(num2str(p_per_type'))
end
%%
figure
for tn= 1:3
    subplot(1,3,tn)
    bar(pctg_ones_per_area(:,tn))
    ylim([0 1])
    xticklabels(area_list)
    title(cell_type{tn})
end
%%
alpha_level_corr = 0.05/15;
figure
area_list = {'SupCol','DLPAG','LPAG','VLPAG','DR'};
cell_type = {'non-entrained','peak','trough'};

for an=1:numel(area_list)
    
    area_index              = ismember(all_neurons_TD.area, area_list{an});
    entrainment_index       = all_neurons_TD.DeltaEntireSession.PPCPval<alpha_level;
    non_entrained_index     = all_neurons_TD.DeltaEntireSession.PPCPval>non_entrained_lvl;
    peak_index              = all_neurons_TD.DeltaEntireSession.PreferedAngle>-pi/2 & all_neurons_TD.DeltaEntireSession.PreferedAngle<pi/2;
    analysed                    = ~isnan(all_neurons_TD.Exited);
    

    peak_cells          = analysed  & area_index & entrainment_index & peak_index;
    trough_cells        = analysed  & area_index & entrainment_index & ~peak_index;
    non_entrained_cells = analysed  & area_index & non_entrained_index;
    
    types_cell = {non_entrained_cells,peak_cells,trough_cells};
    
    for tn= 1:3
        subplot(numel(area_list), 3, (an-1)*3  + tn)
        % play_counts = [;inhibited_play(types_cell{tn})];
        % non_play_counts = [;inhibited_non_play(types_cell{tn})];
        % x12corr = exited_play(types_cell{tn});
        % x22corr = inhibited_non_play(types_cell{tn});
         x12corr = inhibited_non_play(types_cell{tn});
        x22corr = exited_play(types_cell{tn});
        rand_x = (rand(numel(x12corr),1))/5;
        rand_y = (rand(numel(x22corr),1))/5;

        % plot((rand_x + ones(numel(exited_play(types_cell{tn})),2)*diag([1 2]))', [exited_play(types_cell{tn}) exited_non_play(types_cell{tn})]', ':k')
        % hold on
        %  plot((rand_x + ones(numel(exited_play(types_cell{tn}) ),2)*diag([1 2]))', [exited_play(types_cell{tn}) exited_non_play(types_cell{tn})]', '.k', 'MarkerSize',5)
        
        plot(x12corr+rand_x, x22corr+rand_y, '.')

            % [p,h] = signrank(exited_play(types_cell{tn}), exited_non_play(types_cell{tn}))
            [c,p] = corr(x12corr,x22corr, 'Type', 'Spearman')
        % swarmchart([play_counts*0;non_play_counts*0+1],[play_counts;non_play_counts],'.k')
        ylim([-1 1])
        if p<alpha_level_corr
        title(num2str([c,p]), 'Color','r')
        else
            title(num2str([c,p]))
        end
        axis([-.2 1 -.2 1])

    end
end


      
%%
c_lim = [-2 2];
% c_lim = 'auto'
figure
alpha = 0.01;
condition = 'Exited';
psth2plot1 = 'PsthWarped';
% psth2plot1 = 'PsthWarpedPvalPOASelf';
psth2plot2 = 'PsthWarped';
% psth2plot2 = 'PsthWarpedPvalPWIASelf'
this_area = {'LPAG'};
subplot(4,1,1)
trough_cells    = all_neurons_TD.DeltaEntireSession.PPCPval<=alpha & (all_neurons_TD.DeltaEntireSession.PreferedAngle>pi/2   | all_neurons_TD.DeltaEntireSession.PreferedAngle<-pi/2);
peak_cells      = all_neurons_TD.DeltaEntireSession.PPCPval<=alpha & ~(all_neurons_TD.DeltaEntireSession.PreferedAngle>pi/2   | all_neurons_TD.DeltaEntireSession.PreferedAngle<-pi/2);
% cond1 = trough_cells & all_neurons_TD.(condition)==1 & ismember(all_neurons_TD.area, this_area);
% cond2 = peak_cells & all_neurons_TD.(condition)==1 & ismember(all_neurons_TD.area, this_area);
cond1 = trough_cells & ismember(all_neurons_TD.area, this_area);
cond2 = peak_cells  & ismember(all_neurons_TD.area, this_area);
matrtix2plot1 = all_neurons_TD.(psth2plot1)(cond1,:);

for j=1:size(matrtix2plot1,1)
    if std(matrtix2plot1(j,baseline_index), 'omitmissing')>0
        matrtix2plot1(j,:) = (matrtix2plot1(j,:) - mean(matrtix2plot1(j,baseline_index), 'omitmissing'))/std(matrtix2plot1(j,baseline_index), 'omitmissing');
    else
         matrtix2plot1(j,:) = NaN;
    end
end

matrtix2plot1(abs(matrtix2plot1)>4) = NaN;

imagesc(warped_time, 1:size(matrtix2plot1,1),matrtix2plot1)
axis xy
clim(c_lim)



subplot(4,1,2)
matrtix2plot2 = all_neurons_TD.(psth2plot2)(cond2,:);
for j=1:size(matrtix2plot2,1)
    if std(matrtix2plot2(j,baseline_index), 'omitmissing')>0
        matrtix2plot2(j,:) = (matrtix2plot2(j,:) - mean(matrtix2plot2(j,baseline_index), 'omitmissing'))/std(matrtix2plot2(j,baseline_index), 'omitmissing');
    else
         matrtix2plot2(j,:) = NaN;
    end
end
matrtix2plot2(abs(matrtix2plot2)>4) = NaN;
imagesc(warped_time, 1:size(matrtix2plot2,1),matrtix2plot2)
axis xy
clim(c_lim)


subplot(4,1,3:4)

[~, ~, ci] = ttest(matrtix2plot2);
no_nan = ~any(isnan(ci));

fill([warped_time fliplr(warped_time(no_nan))],[ci(1,:) fliplr(ci(2,no_nan))], 'k', 'FaceAlpha',.2, 'EdgeColor','none')
hold on
plot(warped_time, mean(matrtix2plot2, 'omitmissing'), 'k' )

[~, ~, ci] = ttest(matrtix2plot1);
no_nan = ~any(isnan(ci));
fill([warped_time fliplr(warped_time(no_nan))],[ci(1,:) fliplr(ci(2,no_nan))], 'r', 'FaceAlpha',.2, 'EdgeColor','none')
plot(warped_time, mean(matrtix2plot1, 'omitmissing'), 'r' )



%%

c_lim = [-2 2];
% c_lim = 'auto'
figure
this_area = {'SupCol'}
so =1;
for self_other = {'Self','Other' }
    if ismember('Self', self_other)
    behavior_list = {'PsthWarped','PsthWarpedCHSelf','PsthWarpedEVSelf','PsthWarpedESSelf','PsthWarpedCDSelf','PsthWarpedCBSelf','PsthWarpedPOASelf', ...
        'PsthWarpedPOBSelf','PsthWarpedPWIASelf','PsthWarpedPWIBSelf','PsthWarpedGRSelf','PsthWarpedSNSelf','PsthWarpedRESelf'}

    else

    behavior_list = {'PsthWarped','PsthWarpedCHOther','PsthWarpedEVOther','PsthWarpedESOther','PsthWarpedCDOther','PsthWarpedCBOther','PsthWarpedPOAOther', ...
        'PsthWarpedPOBOther','PsthWarpedPWIAOther','PsthWarpedPWIBOther','PsthWarpedGROther','PsthWarpedSNOther','PsthWarpedREOther'}
    end
    behavior_labels = {'Play Bout','Chasing','Evasion','Escape','Darting','Playful Approach','Pounce Neck','Pounce Back','PWI (neck)','PWI (back)', 'Grooming','Sniffing','Rearing'}
    for   bn = 1:numel(behavior_list)
        condition = 'Exited';
        psth2plot1 = behavior_list{bn};

        trough_cells = all_neurons_TD.ThetaEntireSession.PPCPval<0.05 & (all_neurons_TD.ThetaEntireSession.PreferedAngle>pi/2 | all_neurons_TD.ThetaEntireSession.PreferedAngle<pi/2);

        subplot(6,numel(behavior_list),bn + (so-1)*numel(behavior_list)*3)
        matrtix2plot1 = all_neurons_TD.(psth2plot1)( ismember(all_neurons_TD.area, this_area) & trough_cells,:);

        for j=1:size(matrtix2plot1,1)
            if std(matrtix2plot1(j,baseline_index), 'omitmissing')>0
                matrtix2plot1(j,:) = (matrtix2plot1(j,:) - mean(matrtix2plot1(j,baseline_index), 'omitmissing'))/std(matrtix2plot1(j,baseline_index), 'omitmissing');
            else
                matrtix2plot1(j,:) = NaN;
            end
        end

        matrtix2plot1(abs(matrtix2plot1)>4) = NaN;

        imagesc(warped_time, 1:size(matrtix2plot1,1),matrtix2plot1)
        axis xy
        clim(c_lim)
        title([behavior_labels{bn}, ' ', self_other{1}])





        subplot(6,numel(behavior_list),numel(behavior_list)*[1 2]+bn + (so-1)*numel(behavior_list)*3)
        hold on

        [~, ~, ci] = ttest(matrtix2plot1);
        no_nan = ~any(isnan(ci));
        fill([warped_time fliplr(warped_time(no_nan))],[ci(1,:) fliplr(ci(2,no_nan))], 'r', 'FaceAlpha',.2, 'EdgeColor','none')
        plot(warped_time, mean(matrtix2plot1, 'omitmissing'), 'r' )
        ylim([-2 2])
        % title(sum(ismember(all_neurons_TD.area, this_area) & trough_cells)/sum(ismember(all_neurons_TD.area, this_area)))

    end
    so = so+1;
    sgtitle(this_area)



end

