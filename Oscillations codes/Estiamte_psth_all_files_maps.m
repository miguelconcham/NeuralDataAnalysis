

npx_Raw_Data = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\NPX data\NPX raw data';
saving_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\Theta psth';
animal_list = dir(npx_Raw_Data);
animal_list(1:2) = [];


animal_file_names =  cellfun(@(x) ['B', x],strsplit([animal_list.name], 'B'), 'UniformOutput',false)';
animal_file_names(1) = [];
% animal2exclude = {'B4D4 0826 Dual'};
animal2exclude = {''};
animal_list(ismember(animal_file_names,animal2exclude)) = [];
animal_list = animal_list(2:3);
animal_names ={};
% n_strctut = 1;

% psth_structure = [];
% wind_length     = 1;
% wind_overlap    = .990;
% min_separation = .200;
% f               = .1:.1:6;
% freq_pow_range  = [.5 5];
% 
% psth_structure = [];
wind_length     = .250;
wind_overlap    = .240;
min_separation = .200;
f               = 5:.1:14;
freq_pow_range  = [6 12];

%%
for fn = 2:numel(animal_list)

    if fn==1
        psth_structure = GENERATE_THETA_PSTH_MAPS([npx_Raw_Data, '\', animal_list(fn).name],wind_length,wind_overlap,min_separation,f,freq_pow_range )
        n_strctut = n_strctut+numel(psth_structure);
        animal_names = [animal_names;[repmat(animal_list(fn).name,numel(psth_structure),1) num2cell(1:numel(psth_structure))']]
    else
        transt_psth = GENERATE_THETA_PSTH_MAPS([npx_Raw_Data, '\', animal_list(fn).name],wind_length,wind_overlap,min_separation,f,freq_pow_range )
      
        for sub_j=1:numel(transt_psth)
    
            psth_structure(n_strctut) = transt_psth(sub_j);
            n_strctut = n_strctut+1;
        end
        animal_names = [animal_names;[repmat({animal_list(fn).name},numel(transt_psth),1) num2cell(1:numel(transt_psth))' ]]

    end


end
%% save if needed
disp('saving')
save([saving_folder,'\psth_structure_delta_map_corrected_files.mat'],'psth_structure', '-v7.3');
save([saving_folder,'\animal_names_delta_map_corrected_files.mat'],'animal_names');

%% load if needed
disp('loading')
saving_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Analysis results\Theta psth';

load([saving_folder,'\psth_structure_delta_map.mat'],'psth_structure');
load([saving_folder,'\animal_names_delta_map.mat'],'animal_names');

psth_structure_all_files = psth_structure;
animal_names_all_files = animal_names;
load([saving_folder,'\psth_structure_delta_map_corrected_files.mat'],'psth_structure');
load([saving_folder,'\animal_names_delta_map_corrected_files.mat'],'animal_names');
disp('ready')
%% merging_psth
smooth_wind     = 20;
baseline_range  = [-2 0];
animal_label    = {'B1D1','B1S3','B2S2','B3D2', 'B4S2', 'B4D4'};
PROBES          = {'NPX1','NPX1','NPX1','NPX1','NPX2','NPX2'};
allResults      = [];
mean_response_per_area = [];
n_sample = 1;
allResults_persession = [];
global_sn = 1;
animal_index = [];
session_probe_index = [];
for an=1:numel(animal_label)
    electorde_numner = [1 2];
    bin_size = psth_structure(1).wind_length - psth_structure(1).wind_overlap;
    psth_ranges = psth_structure(1).hist_range;
    wrap_range = psth_structure(1).range_time_wrap;
    time = psth_ranges(1):bin_size:psth_ranges(2)+bin_size;
    baseline_index = time<baseline_range(2) & time>baseline_range(1);
    baseline_index_time_wrap = 1:round((abs(wrap_range(1))/bin_size));
    all_psth_onset                  = [];
    all_psth_onset_behavior         = [];
    all_psth_onset_only_playobut    = [];
    all_psth_offset                 = [];
    all_psth_tw                     = [];
    all_psth_tw_3points             = [];
    all_play_bouts          = [];
    time_wrap_time          = [(baseline_index_time_wrap*bin_size) + wrap_range(1),linspace(0,1,psth_structure(1).n_bins_time_wrap),1 + (1:round((abs(wrap_range(2))/bin_size)))*bin_size];
    time_wrap_3_points      = [(baseline_index_time_wrap*bin_size) + wrap_range(1),linspace(0,1-1/psth_structure(1).n_bins_time_wrap,psth_structure(1).n_bins_time_wrap), ...
        linspace(1,2-1/psth_structure(1).n_bins_time_wrap,psth_structure(1).n_bins_time_wrap),2 + (1:round((abs(wrap_range(2))/bin_size)))*bin_size];


   
    session_index = [];
    

    this_animal = animal_label{an};

    animal2merge = find(cell2mat(cellfun(@(x) contains(x, this_animal), animal_names(:,1),'UniformOutput',false)))';
    sess_n = 1;
    for j=animal2merge

        if contains(animal_names{j},animal_label)

            animal_num              = find(cell2mat(cellfun(@(x) contains(animal_names{j},x), animal_label, 'UniformOutput',false)));
            electrode_num           = find(cell2mat(cellfun(@(x) contains(animal_names{j},x), animal_label, 'UniformOutput',false)));
            this_animal_playbouts   = psth_structure(j).play_bouts_table;
            this_animal_lengths     = diff(this_animal_playbouts');
            all_play_bouts = [all_play_bouts;this_animal_playbouts];

            this_psth_onset         = psth_structure(j).play_bout_onset;
            session_index = [session_index;ones(size(this_psth_onset,2),1)*sess_n];
            for ch=1:384
                for trial=1:size(this_psth_onset,2)
                    this_psth_onset(ch,trial,:) = ( this_psth_onset(ch,trial,:) - mean( this_psth_onset(ch,trial,baseline_index)))/std( this_psth_onset(ch,trial,baseline_index));
                    this_psth_onset(ch,trial,:) = movmean(this_psth_onset(ch,trial,:), smooth_wind);
                    this_psth_onset(ch,trial,time> this_animal_lengths(trial)) = NaN;
                end
            end
            all_psth_onset      = cat(2,all_psth_onset, this_psth_onset);

            this_psth_onset     = psth_structure(j).play_bout_onset;
            this_psth_offset    = psth_structure(j).play_bout_offset;
            for ch=1:384
                for trial=1:size(this_psth_offset,2)
                    this_psth_offset(ch,trial,:) = ( this_psth_offset(ch,trial,:) - mean( this_psth_onset(ch,trial,baseline_index)))/std( this_psth_onset(ch,trial,baseline_index));
                    this_psth_offset(ch,trial,:) = movmean(this_psth_offset(ch,trial,:), smooth_wind);
                end
            end
            all_psth_offset = cat(2,all_psth_offset, this_psth_offset);

            this_psth_tw = psth_structure(j).play_bout_tw_this;
            for ch=1:384
                for trial=1:size(this_psth_tw,2)
                    this_psth_tw(ch,trial,:) = ( this_psth_tw(ch,trial,:) - mean( this_psth_tw(ch,trial,baseline_index_time_wrap)))/std( this_psth_tw(ch,trial,baseline_index_time_wrap));
                end
            end
            all_psth_tw = cat(2,all_psth_tw, this_psth_tw);

        sess_n = sess_n+1;

        end
    end

    play_bout_length = diff(all_play_bouts')';


    [sorted_play_bout_length, order] = sort(play_bout_length);




    hard_coded_x_coords = [8 40;258 290; 508 540; 758 790];
    channel_map = psth_structure(animal2merge(1)).channel_map(2:end,:);
    x_pos = channel_map(:,1);
    y_pos = channel_map(:,2);
    response_interval = time>=0 & time<=.5;
    mean_probe_response = squeeze(mean(all_psth_onset(:,:,:),2, 'omitmissing'));

    stdperch = nan(384,1);

    for j =1:384
        stdperch(j ) = std(mean_probe_response(j,:), 'omitmissing');
    end
    too_low = find(stdperch<0.1);

    mean_probe_response(too_low,:) = NaN;
    areas = psth_structure(animal2merge(1)).areas_by_channel(2:end);
    all_ch_resp = smooth(mean(mean_probe_response(:,response_interval),2, 'omitmissing'),10);

    ARRAY1      = channel_map;
    CELLARRAY2  = areas;   
    % areas2reference = ismember(areas,  {'LPAG','VLPAG', 'DLPAG', 'DR'});
    areas2reference   = ismember(areas,  {'LPAG'});


    ARRAY3 = (all_ch_resp - mean(all_ch_resp(areas2reference), 'omitmissing'))/std(all_ch_resp(areas2reference), 'omitmissing');
    nbins = 20;
    probeType = PROBES{an};
    [wVals, wAreas, wY, borders] = ...
        wrap_probe_values(ARRAY1, CELLARRAY2, ARRAY3, nbins, probeType);
    allResults(an).wrappedVals = wVals;
    allResults(an).wrappedAreas = wAreas;
    allResults(an).wrappedY = wY;
    allResults(an).borders = borders;

    session_list = unique(session_index  );
    for sn=1:numel(session_list)
            mean_probe_response_this_session = squeeze(mean(all_psth_onset(:,session_index==session_list(sn),:),2, 'omitmissing'));

        all_ch_resp_this_session = smooth(mean(mean_probe_response_this_session(:,response_interval),2, 'omitmissing'),10);
        ARRAY1      = channel_map;
        CELLARRAY2  = areas;
        ARRAY3 = (all_ch_resp_this_session - mean(all_ch_resp_this_session(areas2reference), 'omitmissing'))/std(all_ch_resp_this_session(areas2reference), 'omitmissing');

        [wVals, wAreas, wY, borders] = ...
            wrap_probe_values(ARRAY1, CELLARRAY2, ARRAY3, nbins, probeType);
        session_probe_index = [session_probe_index;global_sn];
        allResults_persession(global_sn).wrappedVals = wVals;
        allResults_persession(global_sn).wrappedAreas = wAreas;
        allResults_persession(global_sn).wrappedY = wY;
        allResults_persession(global_sn).borders = borders;
        global_sn = global_sn+1;
    end
          





    probe_list = 1;
    if all(ismember(x_pos, hard_coded_x_coords))
        probe_list = find(any(ismember( hard_coded_x_coords,x_pos),2));
    end
    for pb = 1:numel(probe_list)
        index2use = 1:384;
        areas = psth_structure(animal2merge(1)).areas_by_channel(2:end);
        [y_sorted, order] = sort(y_pos(index2use));
        if all(ismember(x_pos, hard_coded_x_coords))
            index2use = index2use(ismember(x_pos,hard_coded_x_coords(probe_list(pb),:)));

            [y_sorted, order] = sort(y_pos(index2use));
            index2use = index2use(order);
            areas = areas(index2use);
        end


        x_lim = [-2 4];



        mean_probe_response = squeeze(mean(all_psth_onset(index2use,:,:),2, 'omitmissing'));
        figure
        subplot(5,5,[1:4,6:9,11:14,16:19])
        imagesc(time,1:numel(index2use),mean_probe_response)
        xlim(x_lim)
        clim([-1 1])
        axis xy
        pag_range = cell2mat(cellfun(@(x) contains(x, 'PAG'), areas, 'UniformOutput',false));
        pag_range = find(pag_range);
        hold on
        beg_ch = min(pag_range);
        end_ch = max(pag_range);
        plot(x_lim, [1 1]*beg_ch, 'w')
        plot(x_lim, [1 1]*end_ch, 'w')
        title([this_animal, ' ', num2str(f(1))])

        subplot(5,5,5:5:20)
        ch_resp = smooth(max(mean_probe_response(:,response_interval),[],2, 'omitmissing'),10);
                % ch_resp = (max(mean_probe_response(:,response_interval),[],2, 'omitmissing'));

        plot(ch_resp, y_sorted)
        ylim([y_sorted(1) y_sorted(end)])

        subplot(5,5,21:24)
        hold on
        area_list= unique(areas);
        session_list = unique(session_index  );


        mean_response_this_probe = nan(numel(area_list),size(mean_probe_response,2));
         mean_response_this_probe_this_session = nan(numel(area_list),numel(session_list),size(mean_probe_response,2));

         for area_n=1:numel(area_list)

             mean_this_area = mean(mean_probe_response(ismember(areas, area_list{area_n}),:));
             mean_response_this_probe(area_n,:) = mean_this_area;
             plot(time,mean_this_area )


             for sn=1:numel(session_list)
                 mean_probe_response_this_session = squeeze(mean(all_psth_onset(:,session_index==session_list(sn),:),2, 'omitmissing'));
                 mean_this_area_this_session    = mean(mean_probe_response_this_session(ismember(areas, area_list{area_n}),:));
                 mean_response_this_probe_this_session(area_n,sn,:)=mean_this_area_this_session;
             end


         end
        mean_response_per_area(n_sample).mean_response      = mean_response_this_probe;
        mean_response_per_area(n_sample).area_list          = area_list;
        mean_response_per_area(n_sample).animal             = animal_label{an};
        mean_response_per_area(n_sample).probe_n            = probe_list(pb);
        mean_response_per_area(n_sample).mean_response_ps   =  mean_response_this_probe_this_session;
        n_sample = n_sample+1;

        xlim(x_lim)
        legend(area_list)
        pause(.1)

    end
end
%%

[alignedVals, alignedAreas, alignedY] = align_by_area(allResults, {'LPAG'});
[valsResampled, yResampled] = resample_segments_simple(alignedVals, alignedY, nbins);
figure; hold on;
stacked_val = []
for i = 1:numel(alignedVals)
    stacked_val = [stacked_val; valsResampled{i}];
    plot(yResampled{i}, valsResampled{i},'DisplayName', animal_label{i});
end



 


xline(0,'k--','Target Area'); % target area centered
xlabel('Relative Wrapped Position');
ylabel('Value');
legend show

%%

[alignedVals, alignedAreas, alignedY] = align_by_area(allResults_persession, {'LPAG', 'SupCol'});
[valsResampled, yResampled] = resample_segments_simple(alignedVals, alignedY, nbins);
figure; hold on;
stacked_val = []
for i = 1:numel(alignedVals)
    stacked_val = [stacked_val; valsResampled{i}];
    plot(yResampled{i}, valsResampled{i});
end



animal_label    = {'B1D1','B1S3','B2S2','B3D2', 'B4S2', 'B4D4'};
sessions2include = [1 0 0 1 1 1 1 1 1 1 1 1]==1;
repeated_measures = [1 1 1 2 2 2 3 3 3 4 5 6]';

 
[n, T] = size(stacked_val(sessions2include,:));

intercepts = nan(T,1);
pvals     = nan(T,1);

for t = 1:T
    % Response variable for this time point
    y = stacked_val(sessions2include,t);
    
    % Put into a table for fitlme
    tbl = table(y, categorical(repeated_measures(sessions2include)), ...
                'VariableNames', {'y','Subject'});
    
    % Fit random intercept model
    if sum(isnan(tbl.y))<1
    lme = fitlme(tbl, 'y ~ 1 + (1|Subject)');
    
    % Extract fixed effect (intercept)
    coefTable = lme.Coefficients;
    
    intercepts(t) = coefTable.Estimate(1);   % intercept estimate
    pvals(t)      = coefTable.pValue(1);
    end
end

xline(0,'k--','Target Area'); % target area centered
xlabel('Relative Wrapped Position');
ylabel('Value');
legend show


figure
% plot(stacked_val', ':k')
hold on
% rand_pool = ~any(isnan(ci));
% fill([yResampled{1}(rand_pool) fliplr(yResampled{1}(rand_pool))], [ci(1,rand_pool) fliplr(ci(2,rand_pool))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
plot(yResampled{1},stacked_val(sessions2include,:), ':k', 'LineWidth',.2)

plot(yResampled{1},intercepts, 'k', 'LineWidth',2)

h = pvals<0.05;

beg_end = [find(diff([0; h; 0])==1) find(diff([0; h; 0])==-1)-1];
median_diff = median(diff(yResampled{1}))/2;
beg_end = yResampled{1}(beg_end) + median_diff*repmat([-1 1], size(beg_end,1),1);



y_lim = ylim;

for j=1:size(beg_end,1)
    fill(beg_end(j,[1 2 2 1]), [-.1 -.1 0 0]+y_lim(2), 'k' )
end

plot([1 1], y_lim, 'k')
plot([2 2], y_lim, 'k')
xticks([.5 1.5 2.5])
xticklabels({'isRt', 'PAG/DR', 'SupCol'})
axis tight

%%

all_areas =[]
for j=1:numel(mean_response_per_area)

all_areas = [all_areas;mean_response_per_area(j).area_list];
end

area_list = unique(all_areas);

response_per_area       = cell(numel(area_list),1);
power_per_area_response = cell(numel(area_list),1);
power_per_area_pctl     = cell(numel(area_list),1);
session_per_area        = cell(numel(area_list),1);
auto_corr_per_area      = cell(numel(area_list),1);
auto_corr_per_area_pctl = cell(numel(area_list),1);
response_time = find(time>=0 & time<=6);
baseline_time = time>=-3 & time<=0;
fs = 1/mean(diff(time));
n_rand = 1000;
n_lags = 400;
for j=1:numel(mean_response_per_area)
    disp(j)
    this_session_responses = (mean_response_per_area(j).mean_response_ps);
    this_session_areas = mean_response_per_area(j).area_list;
    response_per_session = [];
    for k=1:numel(this_session_areas)
        if strcmp(this_session_areas{k}, 'isRT')
            this_session_areas{k} = 'isRt';
        end
        idx = ismember(area_list,this_session_areas(k));
        for sn=1:size(this_session_responses,2)

            response_per_area{idx} = [response_per_area{idx};squeeze(this_session_responses(k,sn,:))'];
            [response_p,f] = pspectrum(squeeze(this_session_responses(k,sn,response_time)),fs);
            response_per_session = [response_per_session;response_p'];
            shufled_response = squeeze(this_session_responses(k,sn,:))';
            rand_pool = find(~isnan(shufled_response));
            n_rand_starting_points = randsample(numel(rand_pool)-numel(response_time),n_rand);

            [r,lags] = xcorr(squeeze(this_session_responses(k,sn,response_time)),squeeze(this_session_responses(k,sn,response_time)),n_lags, 'normalized');
            auto_corr_per_area{idx} = [auto_corr_per_area{idx};r'];
            shuffled_p = nan(n_rand,numel(response_p));
            shuffled_r = nan(n_rand,numel(r));

            for n=1:n_rand
               
                
                % rand_start_index =
                % idnex4nand = rand_pool(randsample( numel(rand_pool),numel(response_time),false));
                idnex4nand = rand_pool(randsample( numel(rand_pool),numel(response_time),false));
                
                [baseline_p,f] = pspectrum(shufled_response(idnex4nand),fs);
                shuffled_p(n,:) = baseline_p;
                idnex4nand = rand_pool(n_rand_starting_points(n):n_rand_starting_points(n)+numel(response_time));
                 this_shuffled_r = xcorr(squeeze(this_session_responses(k,sn,idnex4nand)),squeeze(this_session_responses(k,sn,idnex4nand)),n_lags, 'normalized');
                 shuffled_r(n,:) = this_shuffled_r;
            end
            pctls = zeros(1,numel(response_p));
            for t = 1:numel(response_p)
                % Percentile = proportion of A(:,t) <= B(t)
                pctls(t) = mean(shuffled_p(:,t) <= response_p(t)) * 100;
            end

            pctls_ac = zeros(1,numel(r));
            for t = 1:numel(r)
                % Percentile = proportion of A(:,t) <= B(t)
                pctls_ac(t) = mean(shuffled_r(:,t) <= r(t)) * 100;
            end
            power_per_area_pctl{idx}     =  [power_per_area_pctl{idx};pctls];
            power_per_area_response{idx} = [power_per_area_response{idx};response_p'];
            auto_corr_per_area_pctl{idx}     =  [auto_corr_per_area_pctl{idx};pctls_ac];

            session_per_area{idx}           = [session_per_area{idx};{[mean_response_per_area(j).animal, 'P', num2str(mean_response_per_area(j).probe_n)]}];
        end
    end
end
play_song([],[],[])
%%
x_lim = [-1 2];
figure
for j=1:numel(area_list)
    subplot(3,4,j)
   
    imagesc(time,1:size(response_per_area{j},1),response_per_area{j})
    yticks(1:numel(session_per_area{j}))
    yticklabels(session_per_area{j})
    clim([-.75 .75])
    title(area_list{j})
    xlim(x_lim)
end

figure
for j=1:numel(area_list)
    subplot(3,4,j)
    if size(response_per_area{j},1)>1
         [~, ~, ci] = ttest(response_per_area{j});
        rand_pool = ~any(isnan(ci));
        fill([time(rand_pool) fliplr(time(rand_pool))], [ci(1,rand_pool) fliplr(ci(2,rand_pool))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
        hold on
        plot(time,mean(response_per_area{j}),'k')
    elseif size(response_per_area{j},1)==1
      plot(time,response_per_area{j})  
    end
   
    title(area_list{j})
    xlim(x_lim)
    ylim([-.1 .75])
end



%% plot areas together
% areas2plot = {'LPAG','SupCol','VLPAG', 'DLPAG', 'isRt'};
areas2plot = {'LPAG','SupCol'};
 % areas2plot = {'LPAG','isRt'};
y_lim = [-.1 .5];
x_lim = [-1 2];
areas_indexes = find(ismember(area_list,areas2plot))';
areas_this_pair = session_per_area(areas_indexes);

area1index =all( ismember(areas_this_pair{1},areas_this_pair{2}),2);
area2index =all( ismember(areas_this_pair{2},areas_this_pair{1}),2);

matrix2test = cell(numel(areas_indexes),1);
areaindexes = {area1index,area2index};
figure
hold on
color_list =  generateDistinctColors(numel(areas2plot));
n_col = 1;
for j=areas_indexes
    if size(response_per_area{j},1)>1
         [~, ~, ci] = ttest(response_per_area{j});
         matrix2test{n_col}= response_per_area{j}(areaindexes{n_col},:);
        rand_pool = ~any(isnan(ci));
       % fill([time(no_nan) fliplr(time(no_nan))], [ci(1,no_nan) fliplr(ci(2,no_nan))], color_list(n_col,:), 'FaceAlpha',.25, 'EdgeColor','none', 'HandleVisibility','off')
      
        hold on
        plot(time,mean(response_per_area{j}),'Color', color_list(n_col,:))
    elseif size(response_per_area{j},1)==1
      plot(time,response_per_area{j})  
    end
   
    title(area_list{j})
    xlim(x_lim)
    ylim(y_lim)
    n_col = n_col+1;
end



 repeated_measures = session_per_area{areas_indexes(1)}(area1index);
% repeated_measures =  cellfun(@(x) x(1:end-2),repeated_measures, 'UniformOutput',false);
index2estimate = time>=x_lim(1) & time<=x_lim(2);
array2test = matrix2test{1}-matrix2test{2};
array2test = array2test(:,index2estimate);
[n, T] = size(array2test);
intercepts = nan(T,1);
pvals     = nan(T,1);

for t = 1:T
    % Response variable for this time point
    y = array2test(:,t);
    
    % Put into a table for fitlme
    tbl = table(y, categorical(repeated_measures), ...
                'VariableNames', {'y','Subject'});
    
    % Fit random intercept model
    lme = fitlme(tbl, 'y ~ 1 + (1|Subject)');
    
    % Extract fixed effect (intercept)
    coefTable = lme.Coefficients;
    
    intercepts(t) = coefTable.Estimate(1);   % intercept estimate
    pvals(t)      = coefTable.pValue(1);
end


legend(areas2plot)
plot([0 0], [-.1 .5], 'k', 'HandleVisibility','off')
% [h,p,ci, stats] = ttest(matrix2test{1}-matrix2test{2});


h = pvals<0.01;

beg_end = [find(diff([0; h; 0])==1) find(diff([0; h; 0])==-1)-1];
sub_time = time(index2estimate);
beg_end = sub_time(beg_end);



y_lim = ylim;

for j=1:size(beg_end,1)
    fill(beg_end(j,[1 2 2 1]), [-.1 -.1 0 0]+y_lim(2), 'k' )
end





%%
x_lim = [0 8]
figure
for j=1:numel(area_list)
    subplot(3,4,j)
    imagesc(f,1:size(power_per_area_pctl{j},1),power_per_area_pctl{j})
    yticks(1:numel(session_per_area{j}))
    yticklabels(session_per_area{j})
    clim([0 100])
    title(area_list{j})
    xlim(x_lim)
end


figure
for j=1:numel(area_list)
    subplot(3,4,j)
    if size(power_per_area_pctl{j},1)>1
        [~, ~, ci] = ttest(power_per_area_pctl{j});
        rand_pool = ~any(isnan(ci)) ;
         plot(f,(power_per_area_pctl{j}), ':k')
         hold on
        fill([f(rand_pool)' fliplr(f(rand_pool)')], [ci(1,rand_pool) fliplr(ci(2,rand_pool))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
        
        plot(f,mean((power_per_area_pctl{j}), 'omitmissing'), 'k', 'LineWidth',2)

    elseif size(power_per_area_pctl{j},1)==1
      plot(f,power_per_area_pctl{j})  
    end
   
    title(area_list{j})
    xlim(x_lim)
    ylim([0 100])
    % yscale log
    % ylim([-.1 1])
end


%%

merged_areas = {'LPAG','SupCol'};


area_index = find(ismember(area_list, merged_areas));

all_merged_pow = [];
for j=1:numel(area_index)

    all_merged_pow = [all_merged_pow;power_per_area_response{area_index(j)}];
end

figure

[~, ~, ci] = ttest(all_merged_pow);
rand_pool = ~any(isnan(ci)) ;
plot(f,(all_merged_pow), ':k')
hold on
fill([f(rand_pool)' fliplr(f(rand_pool)')], [ci(1,rand_pool) fliplr(ci(2,rand_pool))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')

plot(f,mean(all_merged_pow), 'k')


%% plot particular area 

figure
x_lim = [-1 2];

j = find(ismember(area_list, 'LPAG'));
 subplot(1,4,1)
   
    imagesc(time,1:size(response_per_area{j}(1:end-1,:),1),response_per_area{j}(1:end-1,:))
    yticks(1:numel(session_per_area{j}(1:end-1)))
    yticklabels(session_per_area{j}(1:end-1))
    clim([-.5 .5])
    title(area_list{j})
    xlim(x_lim)

    subplot(1,4,2)
    if size(response_per_area{j},1)>1
         [~, ~, ci] = ttest(response_per_area{j}(1:end-1,:));
        rand_pool = ~any(isnan(ci));
        plot(time, response_per_area{j}(1:end-1,:), ':k')
        hold on
        fill([time(rand_pool) fliplr(time(rand_pool))], [ci(1,rand_pool) fliplr(ci(2,rand_pool))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')        
        plot(time,mean(response_per_area{j}(1:end-1,:)),'k')
    elseif size(response_per_area{j},1)==1
      plot(time,response_per_area{j})  
    end
   
    title(area_list{j})
    xlim(x_lim)
    ylim([-.1 .75])
    plot([0 0], [-.1 .75], 'r')
x_lim = [0 8]

    subplot(1,4,3)
       matrix2test = power_per_area_pctl{j}(1:end-1,:); 
    imagesc(f,1:size(matrix2test,1),matrix2test)
    yticks(1:numel(session_per_area{j}(1:end-1)))
    yticklabels(session_per_area{j}(1:end-1))
    clim([0 100])
    title(area_list{j})
    xlim(x_lim)




    subplot(1,4,4)
    if size(power_per_area_response{j},1)>1
     
        [~, ~, ci] = ttest(matrix2test);
        rand_pool = ~any(isnan(ci)) ;
         plot(f,matrix2test, ':k')
         hold on
        fill([f(rand_pool)' fliplr(f(rand_pool)')], [ci(1,rand_pool) fliplr(ci(2,rand_pool))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
        
        plot(f,mean(matrix2test), 'k')

    elseif size(power_per_area_response{j},1)==1
      plot(f,matrix2test)  
    end
   
    title(area_list{j})
    xlim(x_lim)

%% plot autocorr

x_lim = [-4 4]
figure
for j=1:numel(area_list)
    subplot(3,4,j)
    imagesc(lags,1:size(auto_corr_per_area_pctl{j},1),auto_corr_per_area_pctl{j})
    yticks(1:numel(session_per_area{j}))
    yticklabels(session_per_area{j})
    clim([0 100])
    title(area_list{j})
    xlim(x_lim)
end


figure
for j=1:numel(area_list)
    subplot(3,4,j)
    if size(auto_corr_per_area_pctl{j},1)>1
        [~, ~, ci] = ttest(auto_corr_per_area_pctl{j});
        rand_pool = ~any(isnan(ci)) ;
         plot(lags,(auto_corr_per_area_pctl{j}), ':k')
         hold on
        fill([lags(rand_pool) fliplr(lags(rand_pool))], [ci(1,rand_pool) fliplr(ci(2,rand_pool))], 'k', 'FaceAlpha',.25, 'EdgeColor','none')
        
        plot(lags,mean((auto_corr_per_area_pctl{j}), 'omitmissing'), 'k', 'LineWidth',2)

    elseif size(auto_corr_per_area_pctl{j},1)==1
      plot(lags,auto_corr_per_area_pctl{j})  
    end
   
    title(area_list{j})
    xlim(x_lim)
    ylim([0 100])
    % yscale log
    % ylim([-.1 1])
end

