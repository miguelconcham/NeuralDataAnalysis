MI_folder = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\PlayBout Analysis\DataSets\Preliminar TWO ANIMALS\MI summary';

files2merge = dir(MI_folder);
files2merge(1:2) = [];
all_mi_t_pctl = [];
all_mi_t = [];
wind_length     = 1;
wind_overlap    = .990;
 spect_sr = round(1/(wind_length-wind_overlap));

time_range = [-5 5];
psth_time =( time_range(1)*spect_sr:time_range(2)*spect_sr)/spect_sr;
t_size      = 200;
T = numel(psth_time);
mi_time = psth_time(1:T-t_size) + .5*t_size/spect_sr;

baseline_correction = mi_time>=-2 & mi_time<0;
% baseline_correction =  mi_time<0;

for j=1:numel(files2merge)

    load([MI_folder, '\',files2merge(j).name], 'mi_t_pctl','mi_t')
    all_mi_t_pctl = [all_mi_t_pctl;mi_t_pctl'];
     mi_t  = ( mi_t  - mean( mi_t (baseline_correction)))/ std( mi_t (baseline_correction));
    all_mi_t = [all_mi_t;mi_t'];
   
end

% hardcoded so far




%%

figure
x_lim =[-2 4];
subplot(5,1,1)

imagesc(mi_time, 1:size(all_mi_t,1),all_mi_t)
clim([-3 3])
xlim(x_lim)
subplot(5,1,2:3)
plot(mi_time,all_mi_t, ':k')
axis xy
xlim(x_lim)
hold on

[h, ~, ci] =ttest(all_mi_t);

fill([mi_time fliplr(mi_time)],[ci(1,:) fliplr(ci(2,:))],'k', 'FaceAlpha',.25, 'EdgeColor','none')
plot(mi_time,mean(all_mi_t), 'k')
y_lim = ylim;

start_end = mi_time([find(diff([0,h,0])==1)' find(diff([0,h,0])==-1)'-1]) ;

for j=1:size(start_end,1)

    fill([start_end(j,[1 2 2 1])],y_lim([1 1 2 2]), 'r', 'FaceAlpha',.25, 'EdgeColor','none')
end
xlim(x_lim)




subplot(5,1,4:5)
plot(mi_time,1-all_mi_t_pctl, ':k')
axis xy
xlim(x_lim)
hold on
plot(mi_time,mean(1-all_mi_t_pctl), 'k')

hold on
plot(x_lim, [0.95 0.95], ':r')
