%% define saving folder
saving_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\Theta psth';
figures_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Codes\Figure codes\Figure 3 Updated';
%% peak and trough diagram
figure
x = (1:2000)/1000;
y_red = sin(2*pi*((1:2000)/1000)-pi/2 );
y_red(y_red<0) = NaN;
y_blue = sin(2*pi*((1:2000)/1000)-pi/2 );
y_blue(y_blue>=0) = NaN;


plot(360*x - 180,y_red, 'r' )
hold on
plot(360*x - 180,y_blue, 'b' )
xticks([- 180 -90 0 90 180 270 360])





%% load delta or theta
% load([saving_folder,'\delta_phase_couplig_structure_updated_with_non_playbouts.mat'],'phase_struct');
% load([saving_folder,'\delta_phase_couplig_animal_names_updated_with_non_playbouts.mat'],'animal_names');

load([saving_folder,'\theta_phase_couplig_structure_updated.mat'],'phase_struct');
load([saving_folder,'\theta_phase_couplig_animal_names_updated.mat'],'animal_names');

% load([saving_folder,'\delta_phase_couplig_structure_updated.mat'],'phase_struct');
% load([saving_folder,'\delta_phase_couplig_animal_names_updated.mat'],'animal_names');


delta_bin_size = 0.01;
phase_prop_names = {'PreferedAngle','MVL','MVLPval','PPC','PPCPval','MeanRate', 'Id'};


all_session_phase_stats = [];
all_session_psth        = [];
entire_recording_psth = [];

all_neurons = [];
for ns = 1:numel(phase_struct)

    all_session_phase_stats = cat(2,all_session_phase_stats, phase_struct(ns).session_phase_stats(1:2,:,:));
    all_session_psth = cat(2,all_session_psth, phase_struct(ns).session_psth(1:2,:,:));
    entire_recording_psth = cat(2,entire_recording_psth, phase_struct(ns).entire_recording_psth);
    entre_session_stats = phase_struct(ns).entire_recording_phase_stats;
    this_session_cluster_info = phase_struct(ns).cluster_info;


    sub_Table =  array2table(squeeze(phase_struct(ns).session_phase_stats(1,:,:)));
    sub_Table.Properties.VariableNames = phase_prop_names;
    this_session_cluster_info.Partner1 =sub_Table;

    sub_Table =  array2table(squeeze(phase_struct(ns).session_phase_stats(2,:,:)));
    sub_Table.Properties.VariableNames = phase_prop_names;
    this_session_cluster_info.Partner2 =sub_Table;

    sub_Table =  array2table(squeeze(phase_struct(ns).play_phase_stats(1,:,:)));
    sub_Table.Properties.VariableNames = phase_prop_names;
    this_session_cluster_info.Play =sub_Table;

    sub_Table =  array2table(squeeze(phase_struct(ns).pre_play_phase_stats(1,:,:)));
    sub_Table.Properties.VariableNames = phase_prop_names;
    this_session_cluster_info.PrePlay =sub_Table;


    sub_Table =  array2table(squeeze(phase_struct(ns).entire_recording_phase_stats(1,:,:)));
    sub_Table.Properties.VariableNames = phase_prop_names;
    this_session_cluster_info.EntireSession =sub_Table;

    this_session_cluster_info.session = repmat(animal_names(ns,1),size(this_session_cluster_info,1),1);

    all_neurons = [all_neurons; this_session_cluster_info];
end
%%
save([saving_folder,'\tehta_all_neurons_v2.mat'],'all_neurons');
%% ploting entrainment per area and psths
psth_edges = phase_struct(1).edges_freq;
psth_centers = .5*(psth_edges(1:end-1) + psth_edges(2:end));


center_index = psth_centers>=-.25 & psth_centers<=.25;
selected_bin_size=0.001;
x_lim = [-.5 .5];
freq2ilter = 40;
c_lim =[-10 10] ;
y_lim = [-5 5]
alpha = 0.01;
figure
area_list = {'SupCol' 'DLPAG'	'LPAG'	'VLPAG' 'DR' 'isRt'};

modulation_range_entrained = cell(numel(area_list) ,4);
modulation_range_non_entrained = cell(numel(area_list) ,4);


for an=1:numel(area_list)
    area_index   = strcmp(all_neurons.area, area_list{an});
    entrained_index = all_neurons.EntireSession.PPCPval<alpha & ~isnan(all_neurons.EntireSession.PPC);

    psth_entrained = squeeze(entire_recording_psth(1,entrained_index & area_index,:));
    if size(psth_entrained,2)==1
        psth_entrained = psth_entrained';
    end
    entrained_angles = all_neurons.EntireSession.PreferedAngle(entrained_index & area_index);
    entrained_mvl = all_neurons.EntireSession.MVL(entrained_index & area_index);



    for j=1:size(psth_entrained,1)
        psth_entrained(j,:) = smooth(psth_entrained(j,:),round((1/freq2ilter)/selected_bin_size));
        psth_entrained(j,:) = 100*(psth_entrained(j,:) - mean(psth_entrained(j,:),'omitmissing'))/mean(psth_entrained(j,:),'omitmissing');
    end

    modulation_range_entrained{an,1} = range(psth_entrained(:,center_index),2);
    modulation_range_entrained{an,2} = entrained_angles;
    modulation_range_entrained{an,3} = entrained_mvl;
    modulation_range_entrained{an,4} = psth_entrained;



    psth_non_entrained = squeeze(entire_recording_psth(1,~entrained_index & area_index,:));
    non_entrained_angles = all_neurons.EntireSession.PreferedAngle(~entrained_index & area_index);
    non_entrained_mvl = all_neurons.EntireSession.MVL(~entrained_index & area_index);
    for j=1:size(psth_non_entrained,1)
        psth_non_entrained(j,:) = smooth(psth_non_entrained(j,:),round((1/freq2ilter)/selected_bin_size));
        psth_non_entrained(j,:) = 100*(psth_non_entrained(j,:) - mean(psth_non_entrained(j,:),'omitmissing'))/mean(psth_non_entrained(j,:),'omitmissing');
    end

    modulation_range_non_entrained{an,1} = range(psth_non_entrained(:,center_index),2);
    modulation_range_non_entrained{an,2} = non_entrained_angles;
    modulation_range_non_entrained{an,3} = non_entrained_mvl;
    modulation_range_non_entrained{an,4} = psth_non_entrained;

    entrained_angles(isnan(entrained_angles)) = 0;
    [sorted_angles, order] = sort(entrained_angles);
    matrix2plot = psth_entrained(order,:);
    subplot(4,numel(area_list) ,an)
    hold on
    y_ticks = 180*sorted_angles/pi;
    imagesc(psth_centers,y_ticks ,matrix2plot)
    hold on
    axis xy
    [pos_90, loc_90]    = min(abs((180*sorted_angles/pi )-90));   
    [pos_270, loc_270]  = min(abs((180*sorted_angles/pi)+90));
    [pos_0, loc_0]       = min(abs((180*sorted_angles/pi)));
    clim(c_lim)
    xlim(x_lim)
    ylim([-180 180])
    plot([psth_centers(1) psth_centers(end)],y_ticks([loc_0 loc_0]), 'w')
    plot([psth_centers(1) psth_centers(end)],y_ticks([loc_270 loc_270]), 'w')
    plot([psth_centers(1) psth_centers(end)],y_ticks([loc_90 loc_90]), 'w')
    yticks([-180 y_ticks([loc_270 loc_0 loc_90])' 180])
    yticklabels({'-180','-90','0','90','180'})
    title(area_list{an})

    subplot(4,numel(area_list) ,numel(area_list) + an)
    selection = max(abs((matrix2plot-repmat(mean(matrix2plot,2),1,size(matrix2plot,2)))./repmat(std(matrix2plot,[],2),1,size(matrix2plot,2))),[],2)<8;

    index = sorted_angles>pi/2 | sorted_angles<-pi/2;
    [~, ~, ci] = ttest(matrix2plot(index & selection,:));
    fill([psth_centers fliplr(psth_centers)], [ci(1,:) fliplr(ci(2,:))], 'b', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(psth_centers, mean(matrix2plot(index & selection,:)), 'b', 'LineWidth',2)


    [~, ~, ci] = ttest(matrix2plot(~index & selection,:));
    fill([psth_centers fliplr(psth_centers)], [ci(1,:) fliplr(ci(2,:))], 'r', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(psth_centers, mean(matrix2plot(~index & selection,:)), 'r', 'LineWidth',2)
    ylim(y_lim)
    xlim(x_lim)
      title({num2str(100*[sum(index) sum(~index)]/numel(index)),num2str([sum(index) sum(~index)])})

    subplot(4,numel(area_list) ,2*numel(area_list) +an)
    hold on
    non_entrained_angles(isnan(non_entrained_angles)) =0;
    [sorted_angles, order] = sort(non_entrained_angles);
    matrix2plot = psth_non_entrained(order,:);

    imagesc(psth_centers, 180*sorted_angles/pi,matrix2plot)

    ylim([-180 180])
    plot([psth_centers(1) psth_centers(end)],[0 0], 'w')
    plot([psth_centers(1) psth_centers(end)],-[90 90], 'w')
    plot([psth_centers(1) psth_centers(end)],[90 90], 'w')
    axis xy
    clim(c_lim)
    yticks([-180 -90 0 90 180])
    xlim(x_lim)

    subplot(4,numel(area_list) ,3*numel(area_list) +an)
    selection = max(abs((matrix2plot-repmat(mean(matrix2plot,2),1,size(matrix2plot,2)))./repmat(std(matrix2plot,[],2),1,size(matrix2plot,2))),[],2)<Inf;
    index = sorted_angles>-pi & sorted_angles<0;
    [~, ~, ci] = ttest(matrix2plot(index & selection,:));
    fill([psth_centers fliplr(psth_centers)], [ci(1,:) fliplr(ci(2,:))], 'b', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(psth_centers, mean(matrix2plot(index & selection,:)), 'b', 'LineWidth',2)
  


    [~, ~, ci] = ttest(matrix2plot(~index & selection,:));
    fill([psth_centers fliplr(psth_centers)], [ci(1,:) fliplr(ci(2,:))], 'r', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(psth_centers, mean(matrix2plot(~index & selection,:)), 'r', 'LineWidth',2)
    ylim(y_lim)
    xlim(x_lim)



    pause(.1)
end
%% plot all area together
figure
psth_edges = phase_struct(1).edges_freq;
psth_centers = .5*(psth_edges(1:end-1) + psth_edges(2:end));


center_index = psth_centers>=-.25 & psth_centers<=.25;
selected_bin_size=0.001;
x_lim = [-.2 .2];
freq2ilter = 40;
c_lim =[-15 15] ;
y_lim = [-15 15]
alpha = 0.01;
area_list = {'SupCol' 'DLPAG'	'LPAG'	'VLPAG' 'DR'};
area_index   = ismember(all_neurons.area, area_list);
entrained_index = all_neurons.EntireSession.PPCPval<alpha & ~isnan(all_neurons.EntireSession.PPC);

psth_entrained = squeeze(entire_recording_psth(1,entrained_index & area_index,:));
if size(psth_entrained,2)==1
    psth_entrained = psth_entrained';
end
entrained_angles = all_neurons.EntireSession.PreferedAngle(entrained_index & area_index);
entrained_mvl   = all_neurons.EntireSession.MVL(entrained_index & area_index);



for j=1:size(psth_entrained,1)
    psth_entrained(j,:) = smooth(psth_entrained(j,:),round((1/freq2ilter)/selected_bin_size));
    psth_entrained(j,:) = 100*(psth_entrained(j,:) - mean(psth_entrained(j,:),'omitmissing'))/mean(psth_entrained(j,:),'omitmissing');
end



psth_non_entrained = squeeze(entire_recording_psth(1,~entrained_index & area_index,:));
non_entrained_angles = all_neurons.EntireSession.PreferedAngle(~entrained_index & area_index);
non_entrained_mvl = all_neurons.EntireSession.MVL(~entrained_index & area_index);
for j=1:size(psth_non_entrained,1)
    psth_non_entrained(j,:) = smooth(psth_non_entrained(j,:),round((1/freq2ilter)/selected_bin_size));
    psth_non_entrained(j,:) = 100*(psth_non_entrained(j,:) - mean(psth_non_entrained(j,:),'omitmissing'))/mean(psth_non_entrained(j,:),'omitmissing');
end


entrained_angles(isnan(entrained_angles)) = 0;
[sorted_angles, order] = sort(entrained_angles);
matrix2plot = psth_entrained(order,:);

subplot(4,1 ,1)
hold on
y_ticks = 180*sorted_angles/pi;
imagesc(psth_centers,y_ticks ,matrix2plot)
hold on
axis xy
[pos_90, loc_90]    = min(abs((180*sorted_angles/pi )-90));
[pos_270, loc_270]  = min(abs((180*sorted_angles/pi)+90));
[pos_0, loc_0]       = min(abs((180*sorted_angles/pi)));
clim(c_lim)
xlim(x_lim)
ylim([-180 180])
plot([psth_centers(1) psth_centers(end)],y_ticks([loc_0 loc_0]), 'w')
plot([psth_centers(1) psth_centers(end)],y_ticks([loc_270 loc_270]), 'w')
plot([psth_centers(1) psth_centers(end)],y_ticks([loc_90 loc_90]), 'w')
yticks([-180 y_ticks([loc_270 loc_0 loc_90])' 180])
yticklabels({'-180','-90','0','90','180'})
% title(area_list)

subplot(4,1 ,2)
selection = max(abs((matrix2plot-repmat(mean(matrix2plot,2),1,size(matrix2plot,2)))./repmat(std(matrix2plot,[],2),1,size(matrix2plot,2))),[],2)<8;

index = sorted_angles>pi/2 | sorted_angles<-pi/2;
[~, ~, ci] = ttest(matrix2plot(index & selection,:));
% ci = prctile(matrix2plot,[5 95]);
fill([psth_centers fliplr(psth_centers)], [ci(1,:) fliplr(ci(2,:))], 'b', 'FaceAlpha',.25, 'EdgeColor','none')
hold on
plot(psth_centers, mean(matrix2plot(index & selection,:)), 'b', 'LineWidth',2)


[~, ~, ci] = ttest(matrix2plot(~index & selection,:));
fill([psth_centers fliplr(psth_centers)], [ci(1,:) fliplr(ci(2,:))], 'r', 'FaceAlpha',.25, 'EdgeColor','none')
hold on
plot(psth_centers, mean(matrix2plot(~index & selection,:)), 'r', 'LineWidth',2)
ylim(y_lim)
xlim(x_lim)
title({num2str(100*[sum(index) sum(~index)]/numel(index)),num2str([sum(index) sum(~index)])})

subplot(4,1 ,3)
hold on
non_entrained_angles(isnan(non_entrained_angles)) =0;
[sorted_angles, order] = sort(non_entrained_angles);
matrix2plot = psth_non_entrained(order,:);

imagesc(psth_centers, 180*sorted_angles/pi,matrix2plot)

ylim([-180 180])
plot([psth_centers(1) psth_centers(end)],[0 0], 'w')
plot([psth_centers(1) psth_centers(end)],-[90 90], 'w')
plot([psth_centers(1) psth_centers(end)],[90 90], 'w')
axis xy
clim(c_lim)
yticks([-180 -90 0 90 180])
xlim(x_lim)

subplot(4,1 ,4)
selection = max(abs((matrix2plot-repmat(mean(matrix2plot,2),1,size(matrix2plot,2)))./repmat(std(matrix2plot,[],2),1,size(matrix2plot,2))),[],2)<Inf;
index = sorted_angles>-pi & sorted_angles<0;
[~, ~, ci] = ttest(matrix2plot(index & selection,:));
fill([psth_centers fliplr(psth_centers)], [ci(1,:) fliplr(ci(2,:))], 'b', 'FaceAlpha',.25, 'EdgeColor','none')
hold on
plot(psth_centers, mean(matrix2plot(index & selection,:)), 'b', 'LineWidth',2)



[~, ~, ci] = ttest(matrix2plot(~index & selection,:));
fill([psth_centers fliplr(psth_centers)], [ci(1,:) fliplr(ci(2,:))], 'r', 'FaceAlpha',.25, 'EdgeColor','none')
hold on
plot(psth_centers, mean(matrix2plot(~index & selection,:)), 'r', 'LineWidth',2)
ylim(y_lim)
xlim(x_lim)

title({num2str(100*[sum(index) sum(~index)]/numel(index)),num2str([sum(index) sum(~index)])})
%%
print(gcf,'-vector','-dsvg',[figures_folder, '\phaselocking delt aall areas together.svg'])

%%

area_list = unique(all_neurons.area);

freq2use        = 'EntireSession'; %options:   ThetaEntireSession DeltaEntireSession
alpha_tresh = 0.01;
an=2;
area_index = ismember(all_neurons.area,area_list{an} );
entrainment_index = all_neurons.(freq2use).PPCPval<alpha_tresh;
this_Area_Angles = all_neurons.(freq2use).PreferedAngle(~isnan(all_neurons.(freq2use).PreferedAngle) & entrainment_index);

figure
polarhistogram(this_Area_Angles, -pi:(pi/16):pi)


[mu, kappa, w, LL] = vm2_mixture_EM(this_Area_Angles+pi, 10000);
LL2 = LL(end);

[theta, kapa_unimodal] = circ_vmpar(this_Area_Angles);
LL1 = sum(log(circ_vmpdf(this_Area_Angles, theta, kapa_unimodal)));

LR = 2*(LL2 - LL1)

%%
% figure_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Codes\Figure codes\Figure 7 Inputs';
% 
% N = numel(this_Area_Angles);
% nBoot = 10000;
% LR_boot = zeros(nBoot,1);
% 
% for b = 1:nBoot
%     boot_sample = circ_vmrnd(theta, kapa_unimodal, N);
% 
%     % Fit 2-component mixture
%     [~, ~, ~, LL_boot] = vm2_mixture_EM(boot_sample, 100, 1e-6, false); % fewer iter for speed
%     LL2_boot = LL_boot(end);
% 
%     % Log-likelihood under unimodal null
%     LL1_boot = sum(log(circ_vmpdf(boot_sample, theta, kapa_unimodal)));
% 
%     LR_boot(b) = 2*(LL2_boot - LL1_boot);
% end
% 
% p_value = mean(LR_boot >= LR);
% 
% save([figure_folder, '\LR_boot.mat'], 'LR_boot')
%% proof that is bymodal

figure_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Codes\Figure codes\Figure 7 Inputs';
% figures_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Codes\Figure codes\Figure 5 Inputs';
load([figure_folder, '\LR_boot.mat'], 'LR_boot')

anged_edges = -pi:(pi/32):pi;
angle_counts = histcounts(this_Area_Angles, anged_edges);
angle_counts = angle_counts/sum(angle_counts);

angle_centers = .5*(anged_edges(1:end-1)+anged_edges(2:end));

mixtured_model  = w .* circ_vmpdf(angle_centers, mu(1)-pi, kappa(1)) + ...
      (1-w) .* circ_vmpdf(angle_centers, mu(2)-pi, kappa(2));
mixtured_model = mixtured_model'/sum(mixtured_model);
figure
plot([angle_centers ,angle_centers+2*pi],[angle_counts angle_counts], 'k')
hold on
plot([angle_centers ,angle_centers+2*pi],[mixtured_model mixtured_model], 'b')


figure
subplot(2,1,1)
histogram(LR_boot, 250, 'FaceColor', 'k', 'EdgeColor', 'none')
y_lim = ylim;
hold on
plot([LR LR],y_lim,'r')


xscale log
xlim([.1 LR+10])

mixtured_model  = w .* circ_vmpdf(anged_edges, mu(1)-pi, kappa(1)) + ...
      (1-w) .* circ_vmpdf(anged_edges, mu(2)-pi, kappa(2));
mixtured_model = mixtured_model'/sum(mixtured_model);

subplot(2,1,2)
polarplot(anged_edges,[angle_counts,angle_counts(1)], 'k' )
hold on
polarplot(anged_edges,mixtured_model, 'r' )
r_lim = rlim;

polarplot([mu(1) mu(1)]-pi,r_lim, 'b' )

polarplot([mu(2) mu(2)]-pi,r_lim, 'r' )
%%

print(gcf,'-vector','-dsvg',[figures_folder, '\two distribution plot.svg'])

%% percentages_per_area
area_list = {'SupCol','DLPAG','LPAG','VLPAG', 'DR'}
number_per_area = nan(numel(area_list),4);
for an=1:numel(area_list)
    area_index = ismember(all_neurons.area,area_list{an} );
    entrainment_index = all_neurons.(freq2use).PPCPval<alpha_tresh;
    this_Area_Angles = all_neurons.(freq2use).PreferedAngle(~isnan(all_neurons.(freq2use).PreferedAngle) & entrainment_index & area_index);

    number_per_area(an,1) = sum(this_Area_Angles>pi/2 | this_Area_Angles<-pi/2);
     number_per_area(an,2) = sum(~(this_Area_Angles>pi/2 | this_Area_Angles<-pi/2));
     number_per_area(an,3) = sum(~entrainment_index & area_index & ~isnan(all_neurons.(freq2use).PreferedAngle));
     number_per_area(an,4) = sum( area_index & ~isnan(all_neurons.(freq2use).PreferedAngle));

end


figure
subplot(4,1,1)
bar(diag(1./number_per_area(:,4))*[number_per_area(:,4)-number_per_area(:,3) number_per_area(:,3)], 'stacked')
legend({'Entrained','Non entrained'})
ylabel('%')
subplot(4,1,2:4)
bar(diag(1./sum(number_per_area(:,1:2),2))*number_per_area(:,1:2), 'stacked')
xticklabels(area_list)
ylabel('%')
legend({'Trough','Peak'})




figure
subplot(4,1,1)
bar([number_per_area(:,4)-number_per_area(:,3) number_per_area(:,3)])
subplot(4,1,2:4)
bar(number_per_area(:,1:2))
xticklabels(area_list)
%%
print(gcf,'-vector','-dsvg',[figures_folder, '\percentage of trough and peak cells per area.svg'])
%% plot modulation strength by angle

x_lim = [-.2 .2];
figure
for an=1:numel(area_list)
    angles              = modulation_range_entrained{an,2};
    angles(isnan(angles)) = 0;
    [sorted_angles, order] = sort(angles);

    modulation_range    = modulation_range_entrained{an,1};
    modulation_range(abs(zscore(modulation_range))>2.5) = NaN;
    angle_bins = -pi:(pi/16):pi;
    angle_bins_centers = .5*(angle_bins(1:end-1) + angle_bins(2:end));

    [~, ~, indx] = histcounts(angles,angle_bins);

    mean_modulation = nan(1,numel(angle_bins)-1);

    for j=1:numel(angle_bins)-1

        this_bin_values = indx==j;
        if ~isempty(this_bin_values)
            mean_modulation(j) = mean(modulation_range(indx==j));
        end
    end

    sorted_modulation = smooth(modulation_range(order),10);

    mean_modulation = fillmissing(mean_modulation, 'linear');
    % polarplot(angle_bins_centers,mean_modulation, 'k')

    subplot(4,numel(area_list) ,an)
    polarplot(sorted_angles,sorted_modulation, 'b')



    subplot(4,numel(area_list) ,numel(area_list) + an )
    polarhistogram(angles,-pi:(pi/8):pi, 'FaceColor','k', 'EdgeColor','none', 'Normalization','percentage')
    rlim([0 40])

    subplot(4,3*numel(area_list) ,[3*numel(area_list)*2+ 3*an + [-2 -1],3*numel(area_list)*3+ 3*an + [-2 -1]])
    matrix2plot = modulation_range_entrained{an,4}(order,:);


    imagesc(psth_centers, 180*sorted_angles/pi,matrix2plot)
    axis xy
    clim(c_lim)
    xlim(x_lim)
    ylim([-180 180])
    hold on
    plot([psth_centers(1) psth_centers(end)],[0 0], 'w')
    plot([psth_centers(1) psth_centers(end)],-[90 90], 'w')
    plot([psth_centers(1) psth_centers(end)],[90 90], 'w')
    if an==1
        yticks([-180 -90 0 90 180])
    else
        yticks([])
    end

    subplot(4,3*numel(area_list),[3*numel(area_list)*2+ 3*an,3*numel(area_list)*3+ 3*an] )
    plot(movmean(modulation_range(order),5, 'omitmissing'),180*sorted_angles/pi, 'k')
    hold on
    ylim tight
    yticks([])
    xlim([0 100])
    ylim([-180 180])
    plot([0 100],[0 0], ':k')
    plot([0 100],-[90 90], ':k')
    plot([0 100],[90 90], ':k')
    pause(.1)


    subplot(4,numel(area_list) ,an)
    [c,p]=circ_corrcl(angles(~isnan(angles) & ~isnan(modulation_range)),modulation_range(~isnan(angles) & ~isnan(modulation_range)));
    % [PPCPval, results] = test_theta_rho_modulation(angles(~isnan(angles) & ~isnan(modulation_range)), modulation_range(~isnan(angles) & ~isnan(modulation_range)), 16, 10000,'increase');
    [PPCPval, results] = test_theta_rho_kernel_multiscale(angles(~isnan(angles) & ~isnan(modulation_range)), modulation_range(~isnan(angles) & ~isnan(modulation_range)), [4 8 16 32], 10000);


    % title({['rho = ' num2str(c)], ['PPCPval = ' num2str(p)]})
    % title({['Theta= ' num2str(180*results.theta_max/pi), ' PPCPval = ' num2str(PPCPval)],...
    %     ['rho = ' num2str(c), ' PPCPval = ' num2str(p)]})
    title({['Theta= ' num2str(180*results.peak_theta/pi), ' PPCPval = ' num2str(PPCPval)],...
        ['rho = ' num2str(c), ' PPCPval = ' num2str(p)]})

    hold on
    plot([results.peak_theta results.peak_theta], [0 60], 'b')
    rlim([0 60])


    pause(.1)


end

%%
print(gcf,'-vector','-dsvg',[figures_folder, '\phaselocking theta angle and modulation.svg'])

%%  now plot partner dynamics


%% ploting rate change
all_neurons.area(ismember(all_neurons.area, {'isRT'})) =     {'isRt'  };
% area_list = unique(all_neurons.area)';

% area_list =  {'4N'	'DLPAG'	'DMPAG'	'DR'	'InfCol'	'LPAG'	'LSD'	'LSI'	'SupCol'	'VLPAG'	'isRt'	'mlf'};

area_list = {'SupCol'  'DLPAG'	'LPAG' 'VLPAG'	'DR' 'isRt'	};
col = strcmp(phase_prop_names, 'MeanRate');
figure
concatenated_rate_differences = [];
mean_rate_differences = [];
d_prime = [];
all_rates = [];
for an=1:numel(area_list)
    subplot(1,numel(area_list),an)
    % index = strcmp(all_neurons.area, area_list{an}) & (all_neurons.Partner1.PPCPval<alpha  & all_neurons.Partner2.PPCPval<alpha & ~isnan(all_neurons.Partner1.PPC + all_neurons.Partner2.PPC)) ...
    %     | (all_neurons.EntireSession.PPCPval<alpha & ~isnan(all_neurons.EntireSession.PPC)) ;
    index = strcmp(all_neurons.area, area_list{an}) &  (all_neurons.EntireSession.PPCPval<alpha & ~isnan(all_neurons.EntireSession.PPC)) ;
    x_jit = (rand(sum(index),2) -.5)*.25;
    matrix2plot = 100*(squeeze(all_session_phase_stats(:,index,col))-repmat(all_neurons.fr(index)',2,1))./repmat(all_neurons.fr(index)',2,1);
    plot((repmat([1 2],sum(index),1)+x_jit)',matrix2plot,':k')
    mean_rate_differences = [mean_rate_differences;mean([diff(matrix2plot)' ones(size(matrix2plot,2),1)*an], 'omitmissing')];


    d_prime = [d_prime;[2*mean(diff(matrix2plot))/sqrt(sum(std(matrix2plot,[],2))) an]];
     all_rates = [all_rates;matrix2plot(:)];
    concatenated_rate_differences = [concatenated_rate_differences;[diff(matrix2plot)' ones(size(matrix2plot,2),1)*an]];
    hold on
    plot((repmat([1 2],sum(index),1)+x_jit)',matrix2plot,'.k', 'MarkerSize',3)

    hold on
    plot([1 2]',mean(matrix2plot,2, 'omitmissing'),'_r', 'MarkerSize',4)
    if all(any(isnan(squeeze(all_session_phase_stats(:,index,col))),1))
        p = 1;
    else
        [p,h,t] = signrank(matrix2plot(1,:), matrix2plot(2,:));
        % [h,p,t] = ttest(matrix2plot(1,:), matrix2plot(2,:))
    end
    ylim([-200 200])
    title([area_list{an},  ' ', num2str(p)])
end
%% save if eneded

print(gcf,'-vector','-dsvg',[figures_folder, '\rate differences stats delta.svg'])

%% ploting entreinment change per area
area_list = unique(all_neurons.area);
alpha= 0.01;
area_list = {'SupCol'  'DLPAG'	'LPAG' 'VLPAG'	'DR' 'isRt'	};

col = strcmp(phase_prop_names, 'PPCPval');
entreinment_per_partner = nan(numel(area_list),6);
figure
for an=1:numel(area_list)
    subplot(1,numel(area_list),an)
    index = strcmp(all_neurons.area, area_list{an});
    x_jit = (rand(sum(index),2) -.5)*.25;
    semilogy((repmat([1 2],sum(index),1)+x_jit)',squeeze(all_session_phase_stats(:,index,col)),':k')
    p1_bool     = all_neurons.Partner1.PPCPval(index)<alpha;
    p2_bool     = all_neurons.Partner2.PPCPval(index)<alpha;
    all_bool    = all_neurons.EntireSession.PPCPval(index)<alpha;
    entreinment_per_partner(an,:) = [sum(p1_bool & p2_bool) sum(p1_bool & ~p2_bool) sum(~p1_bool & p2_bool) sum(~p1_bool & ~p2_bool & all_bool) sum(~p1_bool & ~p2_bool & ~all_bool) sum(index)*sum(index)]/sum(index);
    hold on
    semilogy([0 2.5],[alpha alpha],'r')
    title(area_list{an})
end


%% get phase differnces
all_neurons.area(ismember(all_neurons.area, {'isRT'})) =     {'isRt'  };
% area_list = unique(all_neurons.area)';

% area_list =  {'4N'	'DLPAG'	'DMPAG'	'DR'	'InfCol'	'LPAG'	'LSD'	'LSI'	'SupCol'	'VLPAG'	'isRt'	'mlf'};

area_list = {'SupCol'  'DLPAG'	'LPAG' 'VLPAG'	'DR' 'isRt'	};
y_lim = [10^-6 5]
col = strcmp(phase_prop_names, 'PPC');
figure
concatenated_phase_differences = [];
mean_phase_differences = [];
all_phases = [];
for an=1:numel(area_list)
    subplot(1,numel(area_list),an)
    % index = strcmp(all_neurons.area, area_list{an}) & (all_neurons.Partner1.PPCPval<alpha  & all_neurons.Partner2.PPCPval<alpha & ~isnan(all_neurons.Partner1.PPC + all_neurons.Partner2.PPC)) ...
    %     | (all_neurons.EntireSession.PPCPval<alpha & ~isnan(all_neurons.EntireSession.PPC)) ;
    index = strcmp(all_neurons.area, area_list{an}) &  (all_neurons.EntireSession.PPCPval<alpha & ~isnan(all_neurons.EntireSession.PPC)) ;

    x_jit = (rand(sum(index),2) -.5)*.25;
    matrix2plot = squeeze(all_session_phase_stats(:,index,col));
    all_phases = [all_phases;matrix2plot(:)];
    mean_phase_differences = [mean_phase_differences;mean([diff(matrix2plot)' ones(size(matrix2plot,2),1)*an], 'omitmissing')];
    concatenated_phase_differences = [concatenated_phase_differences;[diff(matrix2plot)' ones(size(matrix2plot,2),1)*an]];


    semilogy((repmat([1 2],sum(index),1)+x_jit)',matrix2plot,':k')
    hold on
    semilogy((repmat([1 2],sum(index),1)+x_jit)',matrix2plot,'.k', 'MarkerSize',3)

    semilogy([1 2]',mean(matrix2plot,2, 'omitmissing'),'_r', 'MarkerSize',4, 'LineWidth',3)
    if all(any(isnan(matrix2plot),1))
        p = 1;
    else
        [p,h,t] = signrank(matrix2plot(1,:)',matrix2plot(2,:)');
        % [h,p] = ttest(matrix2plot(1,:)',matrix2plot(2,:)');
    end
    ylim(y_lim)
    title([area_list{an},  ' ', num2str(p)])
end

%% save if eneded

print(gcf,'-vector','-dsvg',[figures_folder, '\phase differences stats delta.svg'])
%%


figure;
subplot(15,1,1)
bar(entreinment_per_partner(:,6))
xticks(1:numel(area_list))
xticklabels(area_list)
axis tight

subplot(15,1,2:5)
bar(entreinment_per_partner(:,1:5), 'stacked')
xticks(1:numel(area_list))
xticklabels(area_list)
axis tight
legend({'Both Partners','Only Partner1','Only Partner2','Generally PL','Not PL'})

subplot(3,1,2)
swarmchart(concatenated_phase_differences(:,2), concatenated_phase_differences(:,1), 'k.')
hold on
plot(mean_phase_differences(:,2),mean_phase_differences(:,1), '_r')

plot([0 numel(area_list)+1], [0 0], 'r')
xticks(1:numel(area_list))
xticklabels(area_list)
xlim tight
ylim([-.02 .02])



subplot(3,1,3)

swarmchart(concatenated_rate_differences(:,2), concatenated_rate_differences(:,1), 'k.')
hold on
plot(mean_rate_differences(:,2),mean_rate_differences(:,1), '_r')

plot([0 numel(area_list)+1], [0 0], 'r')
xticks(1:numel(area_list))
xticklabels(area_list)
xlim tight
ylim([-100 100])

%%

print(gcf,'-vector','-dsvg',[figures_folder, '\phaselocking partner1 partner2 theta.svg'])

%% correlation between entrainment and rate
no_nan      = ~isnan(concatenated_rate_differences(:,1)) &  ~isnan(concatenated_phase_differences(:,1));
in_range    = abs(concatenated_rate_differences(:,1)  - mean(concatenated_rate_differences(:,1), 'omitmissing'))/std(concatenated_rate_differences(:,1), 'omitmissing')<4 ... 
    & abs(concatenated_phase_differences(:,1)  - mean(concatenated_phase_differences(:,1), 'omitmissing'))/std(concatenated_phase_differences(:,1), 'omitmissing')<4 ;
[c,p] = corr(concatenated_rate_differences(no_nan & in_range,1), concatenated_phase_differences(no_nan & in_range,1));
figure
subplot(1,2,1)
plot(concatenated_rate_differences(no_nan & in_range,1), concatenated_phase_differences(no_nan & in_range,1), 'k.')

title([c p])
xlabel('rate differences')
ylabel('phase  differences')

subplot(1,2,2)

no_nan      = ~isnan(all_rates(:,1)) &  ~isnan(all_phases(:,1));
in_range    = abs(all_rates(:,1)  - mean(all_rates(:,1), 'omitmissing'))/std(all_rates(:,1), 'omitmissing')<4 ... 
    & abs(all_phases(:,1)  - mean(all_phases(:,1), 'omitmissing'))/std(all_phases(:,1), 'omitmissing')<4 ;
plot(all_rates(no_nan & in_range,1), all_phases(no_nan & in_range,1), 'k.')
[c,p] = corr(all_rates(no_nan & in_range,1), all_phases(no_nan & in_range,1));

title([c p])


%% plot selcted neurons: set paramers and folder location


npx_Raw_Data    = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\NPX data\NPX raw data';
saving_folder   = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\Theta psth';
animal_list = dir(npx_Raw_Data);
animal_list(1:2) = [];


animal_file_names =  cellfun(@(x) ['B', x],strsplit([animal_list.name], 'B'), 'UniformOutput',false)';
animal_file_names(1) = [];
% animal2exclude = {'B4D4 0826 Dual'};
animal2exclude = {''};
animal_list(ismember(animal_file_names,animal2exclude)) = [];
animal_names ={};
n_strctut = 1;

freq_range_1    = [1 5];
freq_range_2    = [6 12];
sr              = 2500;
filter_order    = 2000;


phase_struct = [];
% Parameters for delta
Hd_freq = designfilt('bandpassfir', ...
    'FilterOrder', filter_order, ...
    'CutoffFrequency1', freq_range_1(1), ...
    'CutoffFrequency2', freq_range_1(2), ...
    'SampleRate', sr, ...
    'DesignMethod', 'window', ...
    'Window', 'hamming');
bin_size_freq = 0.01;

% Parameters for theta
% Hd_freq = designfilt('bandpassfir', ...
% 'FilterOrder', filter_order, ...
% 'CutoffFrequency1', freq_range_2(1), ...
% 'CutoffFrequency2', freq_range_2(2), ...
% 'SampleRate', sr, ...
% 'DesignMethod', 'window', ...
% 'Window', 'hamming');
% bin_size_freq = 0.001;
%% possible sessions to chose
alpha = 0.01;
area2select = 'SupCol';
area_index   = strcmp(all_neurons.area, area2select);
entrained_index = all_neurons.EntireSession.PPCPval<alpha;
session_list = unique(all_neurons.session(area_index & entrained_index));
max_mvl_table = cell(numel(session_list),5);
id_loked = {};

figure
for ss=1:numel(session_list)
    subplot(4,3,ss)

    session_index= find(strcmp(all_neurons.session, session_list(ss)));
    [ max_mvl, loc] = max(all_neurons.EntireSession.PPC(session_index));

    max_mvl_table(ss,:) = {max_mvl,all_neurons.cluster_id(session_index(loc)),all_neurons.EntireSession.MeanRate(session_index(loc)),session_list{ss},session_index(loc)};
    plot(psth_centers,squeeze(entire_recording_psth(1,session_index(loc),:))/bin_size_freq, 'k')
end
max_mvl_table= cell2table(max_mvl_table);
max_mvl_table.Properties.VariableNames = {'PPC','ClusterID','MeanRate','Session', 'PsthRawIndex'};
disp(max_mvl_table)



%% estimate neurons

row2select = 1;
plot_bool = true;
this_neuron_phase_struct = GENERATE_PHASE_COUPLING_NEURON_ID([npx_Raw_Data, '\', max_mvl_table.Session{row2select}],Hd_freq,bin_size_freq , max_mvl_table.ClusterID(row2select), plot_bool );
%%
ylim tight
print(gcf,'-vector','-dsvg',[figures_folder, '\neuron example psth vectorized.svg'])

%%
y_lim = [-2000 2000];
edges_tn = this_neuron_phase_struct.edges_freq;
psth_centers_tn = (edges_tn(2:end) +edges_tn(1:end-1))/2;
x_lim = [-.5 .5];
figure
colormap(1-gray)
subplot(5,1,1:2)
peak_delta_psth = this_neuron_phase_struct.all_psth{1,1};
imagesc(psth_centers_tn,1:size(peak_delta_psth,1),peak_delta_psth)
xlim(x_lim)
axis xy
yticks([])
clim([0 1])


subplot(5,1,3:5)
plot(psth_centers_tn,mean(peak_delta_psth)/bin_size_freq, 'k')
xlim(x_lim)

%%
print(gcf,'-vector','-dsvg',[figures_folder, '\neuron example psth.svg'])

%% now plot neurons

figure
x_lim = [286 292];
subplot(2,5,1:4)
hold off
real_lfp = this_neuron_phase_struct.all_lfp{1,1}-mean(this_neuron_phase_struct.all_lfp{1,1});
real_lfp = smooth(real_lfp, (1/50)*2500);
real_time = this_neuron_phase_struct.all_lfp{1,3};
lfp_phases = this_neuron_phase_struct.all_lfp{1,4};
this_neuron_spikes = this_neuron_phase_struct.all_spikes{1}';
this_neuron_phases = this_neuron_phase_struct.all_phases{1,1};

index2plot_lfp  = real_time>=x_lim(1) & real_time<=x_lim(2);
index2plot_spikes = this_neuron_spikes>=x_lim(1) & this_neuron_spikes<=x_lim(2);
filtered_lfp = this_neuron_phase_struct.all_lfp{1,2}-mean(this_neuron_phase_struct.all_lfp{1,2});
plot(real_time(index2plot_lfp),real_lfp(index2plot_lfp) , 'k')
hold on

plot(real_time(index2plot_lfp), filtered_lfp(index2plot_lfp), 'r', 'LineWidth',2)
plot([this_neuron_spikes(index2plot_spikes);this_neuron_spikes(index2plot_spikes)],y_lim, 'b')
xlim(x_lim)
dela_peak_times     = this_neuron_phase_struct.all_loc_times{1,1};
delta_peak_indexes  = ismember(real_time,dela_peak_times);
delta_peak_phases = lfp_phases(delta_peak_indexes);

subplot(2,5,5)
polarhistogram(lfp_phases, -pi:(pi/32):pi, 'FaceColor', 'k','EdgeColor','none', 'Normalization','percentage')
hold on
polarhistogram(this_neuron_phases, -pi:(pi/32):pi, 'FaceColor', 'r', 'EdgeColor','none', 'Normalization','percentage')
hold on
r_lim = rlim;
polarplot([1 1]*circ_mean(delta_peak_phases'), r_lim,  'b')



sub_time = real_time(index2plot_lfp);
dela_peak_times_this_range = dela_peak_times(dela_peak_times>=x_lim(1) & dela_peak_times<=x_lim(2));
positive_phases =filtered_lfp(index2plot_lfp);
negative_indexes = lfp_phases(index2plot_lfp)<-pi/2 | lfp_phases(index2plot_lfp)>pi/2;
positive_phases( negative_indexes) = NaN;

negative_phases = filtered_lfp(index2plot_lfp);
negative_phases(~negative_indexes) = NaN;




subplot(2,5,5 + (1:4))

plot(sub_time,negative_phases, 'b')
hold on
plot(sub_time,positive_phases, 'r')
y_lim = [-1000 1000];
plot([dela_peak_times_this_range;dela_peak_times_this_range], y_lim, 'r')

subplot(2,5,10)
polarhistogram(delta_peak_phases, -pi:(pi/32):pi,'FaceColor', 'r', 'EdgeColor','none')


%%
print(gcf,'-vector','-dsvg',[figures_folder, '\neuron example lfp and spikes.svg'])


