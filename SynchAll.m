%% create video structure if is not available

framenumber = 1050;
max_dur = Inf;
current_folder = cd;
video_structure =TTL_STRACTION(current_folder,'avi', framenumber, max_dur);

save('video_structure','video_structure')
%%
load video_structure
 TTL_struct = video_structure;
 
 ttl_length = 0.33;
% TTL_struct.FrameRate =14.995561313851100
% TTL_struct.FrameRate =30

frame_rate = TTL_struct.FrameRate;
ttl_n_samples = ttl_length*frame_rate;
light_intensity_handle = figure('units','normalized','outerposition',[0 0 1 1]);
% light_intensity = (diff(TTL_struct.light_intensity));
% time = (1:numel(TTL_struct.light_intensity)-1)/TTL_struct.FrameRate;

light_intensity = TTL_struct.light_intensity -  TTL_struct.light_intensity_ring;
light_intensity(isnan(light_intensity)) = 0;
light_intensity = light_intensity-movmin(light_intensity,2*ttl_n_samples);
time = (1:numel(TTL_struct.light_intensity))/TTL_struct.FrameRate;
plot(time, light_intensity)
% ylim([0 nanmedian(light_intensity)+nanstd(light_intensity)])

xlabel('Time (s)')
ylabel('Relative Light Change')
set(gca, 'FontSize', 18)




%% Select start point, end point and treshold
% subplot(1,2,2)
% histogram(abs(diff(TTL_struct.light_intensity)), 0:.1:255)
% xlim([-25 ((numel(light_intensity)-1)/TTL_struct.FrameRate + 25)])

input_values = ginput(2);
beg_time    = max(min(input_values(:,1)), min(time));
end_time    = min(max(input_values(:,1)), max(time));
threshold   = max(input_values(:,2));

% treshold    = 400;

hold on
plot([1 1]*beg_time, [min(light_intensity) max(light_intensity)], 'g', 'LineWidth',3)
plot([1 1]*end_time, [min(light_intensity) max(light_intensity)], 'r', 'LineWidth',3)
plot([beg_time(1) end_time(end)],[1 1]*threshold, 'c', 'LineWidth',3)

legend({'luminance values', 'begining of detection', 'end of detection', 'threshold'}, 'Location',  'bestoutside')


events_points = find(light_intensity>threshold);
events_points = events_points(events_points>min(find(time>=beg_time)) & events_points<min(find(time>=end_time)));
event_end = [events_points(find(diff(events_points)>1));events_points(end)];
event_start = events_points([1;find(diff(events_points)>1)+1]);
hold on


interp_intersection = nan(numel(event_start),3);

for ev_idx = 1:numel(event_start)

    interp_intersection(ev_idx,1) = interp1([light_intensity(event_start(ev_idx)-1) light_intensity(event_start(ev_idx))],[time(event_start(ev_idx)-1) time(event_start(ev_idx))],threshold);
    interp_intersection(ev_idx,2) = interp1([light_intensity(event_end(ev_idx)) light_intensity(event_end(ev_idx)+1)],[time(event_end(ev_idx)) time(event_end(ev_idx)+1)],threshold);
    interp_intersection(ev_idx,3) = std(light_intensity(event_start(ev_idx):event_end(ev_idx)));
    interp_intersection(ev_idx,4) = mean(light_intensity(event_start(ev_idx):event_end(ev_idx)));

end
plot(interp_intersection(:,[1 2]), threshold*[1 1], 'k', 'Linewidth', 3, 'HandleVisibility', 'off')
%% Select real ttl (change axis if needed)

figure('units','normalized','outerposition',[0 0 1 1]);

plot(interp_intersection(:,2) - interp_intersection(:,1), interp_intersection(:,3), '.')


%axis([0.3 0.4 0 20])
xlabel('Video TTL duration (s)')
ylabel('Light intensity STD during TTL')
numberOfpoints = 8;
area_point_1 =  ginput(numberOfpoints);
hold on
plot([area_point_1(:,1);area_point_1(1,1)], [area_point_1(:,2); area_point_1(1,2)], 'g', 'LineWidth',3)

cluster_1= inpolygon(interp_intersection(:,2) - interp_intersection(:,1),interp_intersection(:,3),area_point_1(:,1),area_point_1(:,2))
plot(interp_intersection(cluster_1,2) - interp_intersection(cluster_1,1), interp_intersection(cluster_1,3), 'g.')

area_point_2 =  ginput(numberOfpoints);
hold on
plot([area_point_2(:,1);area_point_2(1,1)], [area_point_2(:,2); area_point_2(1,2)], 'r', 'LineWidth',3)

cluster_2 =inpolygon(interp_intersection(:,2) - interp_intersection(:,1),interp_intersection(:,3),area_point_2(:,1),area_point_2(:,2));
plot(interp_intersection(cluster_2,2) - interp_intersection(cluster_2,1), interp_intersection(cluster_2,3), 'r.')


hold on

figure(light_intensity_handle)
hold on
if sum(cluster_1)>0
plot(interp_intersection(cluster_1,[1 2]), threshold*[1 1], 'g', 'Linewidth', 3, 'HandleVisibility', 'off')
end
if sum(cluster_2)>0
plot(interp_intersection(cluster_2,[1 2]), threshold*[1 1], 'r', 'Linewidth', 3, 'HandleVisibility', 'off')
end

excluded_list = find(~(cluster_1|cluster_2));

y_lim = ylim;

for j=excluded_list'

    fill([interp_intersection(j,1)  interp_intersection(j,2) interp_intersection(j,2)  interp_intersection(j,1) ], y_lim([1 1 2 2]), 'k', 'FaceAlpha',.5, 'EdgeColor', 'none', 'HandleVisibility','off')
end


% interp_intersection = interp_intersection(ttl_in_range,:);

%% Exctact TTLs from Npx based on old audio code


% 
% NPX_start_time = get_synch_time(0, 'ProbeA');
% if isnan(NPX_start_time)
%     if exist('synch_timestamps.npy','file')>0
%         timestamps_cont = readNPY('timestamps.npy');
%         NPX_start_time = timestamps_cont(1);
%     elseif exist('synchronized_timestamps.npy','file')>0
%         timestamps_cont = readNPY('synchronized_timestamps.npy');
%         NPX_start_time = timestamps_cont(1);
% 
%     end
%     timestamps_cont = double(NPX_start_time);
%     disp(['START TIME in timestapms at ', num2str(NPX_start_time/30000)])
% end

NPX_start_time          = double( 5685802)/30000 ;



%channel_states          = readNPY([path 'channel_states.npy']);
% path = '\\experimentfs\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\20240826\2024-08-26_14-48-45\Record Node 101\experiment1\recording1\events\Neuropix-PXI-100.ProbeA\TTL\'



timestamps              = readNPY( 'timestamps.npy');
if isa(timestamps, 'double')
    channel_states          = readNPY('states.npy');
else
    channel_states          = readNPY('channel_states.npy');
end
if isa(timestamps, 'double')
      TTLS_Npxs               = [timestamps(channel_states==1) timestamps(channel_states==-1)] -NPX_start_time/30000;
else
    TTLS_Npxs               = [double(timestamps(channel_states==-1)) double(timestamps(channel_states==1))]/30000  -NPX_start_time/30000;
end
TTLS_Npxs(any(TTLS_Npxs==-1,2),:)=[];
%%
audio_files = dir('*.wav');

if ~isempty(audio_files)

TTL_audio_struct = audioTTL(cd)
TTL_audio = [TTL_audio_struct.TTL_Up_in_s, TTL_audio_struct.TTL_Down_in_s];
else
    synch_model_audio2NPX = NaN;
end
    
%% 

TTL_video = interp_intersection(cluster_1|cluster_2,:);

%% Patch for 311023 and 25082024, video is far shorter than Npx

% TTL_audio=TTL_audio(1:size(TTL_video,1),:);

% TTL_video=TTL_video(TTL_video(:,1)>TTL_audio(2),:);
% TTL_audio=TTL_audio(1:size(TTL_video,1));

%% find start

detection_plot = figure('units','normalized','outerposition',[0 0 1 1]);
plot(time, light_intensity)
hold on
plot(interp_intersection(cluster_1,[1 2]), threshold*[1 1], 'g', 'Linewidth', 3, 'HandleVisibility', 'off')
% plot(interp_intersection(cluster_2,[1 2]), threshold*[1 1], 'r', 'Linewidth', 3, 'HandleVisibility', 'off')

ttl_match_number = 10;
time_search_npxs = [0 120];

current_error   = 0.01;

audio_pos = min(find(TTLS_Npxs(:,1)>time_search_npxs(1)));


ttl_distance_video = diff(mean(TTL_video(:,[1 2]),2));
ttl_distance_npx = diff(mean(TTLS_Npxs,2));
video_pos = 1;


error_tolerance = 0.01;
min_error = 0.01;
max_iter = 300;

while min_error>=error_tolerance
    [min_pos,min_error] = find_match(ttl_distance_video,ttl_distance_npx,ttl_match_number, 1,audio_pos,max_iter);
    max_iter = max_iter+5;
end

disp(['Min distance found = ' num2str(min_error) ' at NPX = ' num2str(TTLS_Npxs(min_pos(1)))  's & VIDEO = ' num2str(TTL_video(min_pos(2),1)) 's'])


hold on
time_range = [TTL_video(min_pos(2),1)-1 TTL_video(min_pos(2)+ttl_match_number-1,1)+1];
xlim(time_range)
y_lim = [0 (max(TTL_video(:,4)) + 2*max(TTL_video(:,3)))];
ylim(y_lim)

for j=1:(ttl_match_number-1)
    beg_time = TTL_video(min_pos(2),1) + sum(ttl_distance_npx(min_pos(1):min_pos(1)+j-1));
    end_time = TTL_video(min_pos(2),2) + sum(ttl_distance_npx(min_pos(1):min_pos(1)+j-1));
    fill([beg_time  end_time end_time beg_time], y_lim([1 1 2 2]), 'y', 'FaceAlpha',.5, 'EdgeAlpha',.5, 'HandleVisibility', 'off')
end

title(min_pos(2)+(ttl_match_number-1))






Starting_ttl = min_pos +  ttl_match_number-1;
% ttl_match_number = [TTLS_Npxs(min_pos(1):(min_pos(1)+4),1) interp_intersection(min_pos(2):(min_pos(2)+4), 1) ];
save('ttl_match', 'ttl_match_number')



%% Find other TTLS: set parametters
tolerancePs =5/frame_rate ; %This has to be significantly shorter that the inter TTL interval!!!!!!
PastFut = [-5 5];
figure(detection_plot)
hold on

xlim(time_range)

current_video_ttl  = Starting_ttl(2);
current_npx_ttl     = Starting_ttl(1);

time_range = TTL_video(current_video_ttl) + PastFut;
xlim(time_range)

transitory_math = [];
gap = 0;
ttl_distance_video = diff(TTL_video(:,1));
ttl_distance_npx    = diff(TTLS_Npxs(:,1));
max_gap = 20;

TTL_MATCH =  [(min_pos(1):(min_pos(1)+ ttl_match_number-1))', (min_pos(2):(min_pos(2)+ ttl_match_number-1))', ones(ttl_match_number,1)*-2];
%% MATCHIN TTLS

ttl_match_number= 5;
min_error       = 2*tolerancePs;
error_tolerance = 2*tolerancePs;
max_iter        = 5;
% while current_video_ttl<=numel(TTL_video)


while current_video_ttl<size(TTL_video,1)
    beg_time = TTL_video(current_video_ttl,1) + sum(ttl_distance_npx(current_npx_ttl:(current_npx_ttl+gap)));
    end_time = TTL_video(current_video_ttl,2) + sum(ttl_distance_npx(current_npx_ttl:(current_npx_ttl+gap)));
    mean_time = mean(TTL_video(current_video_ttl,[1 2]),2) + sum(ttl_distance_npx(current_npx_ttl:(current_npx_ttl+gap)));
    error = abs(mean_time - mean(TTL_video(current_video_ttl+1,[1 2]),2));
    if abs(mean_time - mean(TTL_video(current_video_ttl+1,[1 2]),2))<tolerancePs*ttl_distance_npx(current_npx_ttl)
        fill([beg_time  end_time end_time beg_time], y_lim([1 1 2 2]), 'y', 'FaceAlpha',.5, 'EdgeAlpha',.5, 'HandleVisibility', 'off')
        TTL_MATCH = [TTL_MATCH;[current_npx_ttl current_video_ttl error]];

        current_video_ttl  = current_video_ttl+1;
        current_npx_ttl = current_npx_ttl+1+gap;
        time_range = beg_time + PastFut;
        xlim(time_range)
        title(current_video_ttl)
        gap=0;
    elseif gap<max_gap

        fill([beg_time  end_time end_time beg_time], y_lim([1 1 2 2]), 'r', 'FaceAlpha',.25, 'EdgeAlpha',.5, 'HandleVisibility', 'off');
        time_range = beg_time + PastFut;
        xlim(time_range)
        gap = gap+1;
    else
        disp('Gap too long, probable frame lost, searching for ttl match')
        
        while min_error>=error_tolerance
            [min_pos,min_error] = find_match(ttl_distance_video,ttl_distance_npx,ttl_match_number, current_video_ttl,current_npx_ttl,max_iter);
            max_iter = max_iter+1;
        end

        for j=1:(min_pos(1)-current_npx_ttl)
            beg_time = TTL_video(min_pos(2),1) + sum(ttl_distance_npx(min_pos(1):min_pos(1)+j-1));
            end_time = TTL_video(min_pos(2),2) + sum(ttl_distance_npx(min_pos(1):min_pos(1)+j-1));
            fill([beg_time  end_time end_time beg_time], y_lim([1 1 2 2]), 'r', 'FaceAlpha',.5, 'EdgeAlpha',.5, 'HandleVisibility', 'off')
        end
        for j=1:(ttl_match_number-1)
            beg_time = TTL_video(min_pos(2),1) + sum(ttl_distance_npx(min_pos(1):min_pos(1)+j-1));
            end_time = TTL_video(min_pos(2),2) + sum(ttl_distance_npx(min_pos(1):min_pos(1)+j-1));
            fill([beg_time  end_time end_time beg_time], y_lim([1 1 2 2]), 'y', 'FaceAlpha',.5, 'EdgeAlpha',.5, 'HandleVisibility', 'off')
        end
        time_range = beg_time + PastFut;
        xlim(time_range)

        TTL_MATCH = [TTL_MATCH;[(min_pos(1):(min_pos(1)+ ttl_match_number-1))', (min_pos(2):(min_pos(2)+ ttl_match_number-1))' ones(ttl_match_number,1)*-1]];

        current_npx_ttl = min_pos(1)+ ttl_match_number-1;
        current_video_ttl = min_pos(2)+ ttl_match_number-1;


        title(current_video_ttl)
    end

    pause(0.01)
end
%% Plot results and save synch (VIDEO 2 NPX)

figure


 data2fit = [mean(TTL_video(TTL_MATCH(:,2),[1 2]),2),mean(TTLS_Npxs(TTL_MATCH(:,1),:),2)];
    diff_data= diff(data2fit);
subplot(1,2,1)
 histogram(TTL_MATCH(TTL_MATCH(:,3)>0,3), 0:0.001:2/frame_rate, 'FaceColor', 'k', 'EdgeColor','none')
 hold on
histogram(abs(diff_data(:,2) -diff_data(:,1)) , 0:0.001:2/frame_rate, 'FaceColor', 'r', 'EdgeColor','none')

 hold on
 y_lim = ylim;
 plot([1/frame_rate 1/frame_rate], ylim, 'r')
 subplot(1,2,2)




 % data2fit = [TTL_video(:,1),TTLS_Npxs];
 data2fit = array2table(data2fit);
 data2fit.Properties.VariableNames = {'VIDEO TTL (s)','NPX TTL (s)'};      
 synch_model_video2NPX = fitlm(data2fit);



 plot(synch_model_video2NPX)

 save('synch_model_video2NPX', 'synch_model_video2NPX');

 %% (AUDIO 2 NPX)
if ~isempty(audio_files)


figure


ttl_distance_audio = diff(mean(TTL_audio(:,[1 2]),2));
ttl_distance_npx = diff(mean(TTLS_Npxs,2));
video_pos = 1;


error_tolerance = 0.01;
min_error = 0.01;
max_iter = 300;

while min_error>=error_tolerance
    [min_pos,min_error] = find_match(ttl_distance_audio,ttl_distance_npx,ttl_match_number, 1,audio_pos,max_iter);
    max_iter = max_iter+5;
end

disp(['Min distance found = ' num2str(min_error) ' at NPX = ' num2str(TTLS_Npxs(min_pos(1)))  's & AUDIO = ' num2str(TTL_video(min_pos(2),1)) 's'])




 data2fit = [mean(TTL_audio(min_pos(1):end,[1 2]),2),mean(TTLS_Npxs(min_pos(2):end,:),2)];
    diff_data= diff(data2fit);
subplot(1,2,1)
 histogram(abs(diff_data(:,1)-diff_data(:,2)), 500, 'FaceColor', 'k', 'EdgeColor','none')
 hold on
histogram(abs(diff_data(:,2) -diff_data(:,1)) , 500, 'FaceColor', 'r', 'EdgeColor','none')

 hold on
 y_lim = ylim;
 plot([1/frame_rate 1/frame_rate], ylim, 'r')
 subplot(1,2,2)




 % data2fit = [TTL_video(:,1),TTLS_Npxs];
 data2fit = array2table(data2fit);
 data2fit.Properties.VariableNames = {'AUDIO TTL (s)','NPX TTL (s)'};      
 synch_model_audio2NPX = fitlm(data2fit);



 plot(synch_model_audio2NPX)


else
    synch_model_audio2NPX = NaN;
    synch_model_video2audio = NaN;
     save('synch_model_video2audio', 'synch_model_video2audio');
end



 save('synch_model_audio2NPX', 'synch_model_audio2NPX');