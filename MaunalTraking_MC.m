% video_loc           = '\\experimentfs.bccn-berlin.pri\experiment\PlayNeuralData\NPX-OPTO PLAY NMM\20231012\_NMMTT_230929 23-10-12 14-32-19.avi';

%ranagetrak = [280 1290];
ranagetrak = [310 1290];

video_loc           = 'control_HC_monitoring_0002 25-06-09 16-09-46.avi';


% 
frame_rate          = 15;
starting_traking    = ranagetrak(1); %up to 605
samplespersec       = 3;
n_frames2track      = ceil(range(ranagetrak)*samplespersec);
%%
starting_frame      = floor(starting_traking*frame_rate);
frames2stract       = starting_frame:(frame_rate/samplespersec):(starting_frame + (frame_rate/samplespersec)*(n_frames2track-1));
vidObj              = VideoReader(video_loc);

traking_values_animal      = nan(n_frames2track,2);
traking_time        = frames2stract/frame_rate;


 %%
% frames2stract = traking_structure.frames2stract;
% traking_values_animal =traking_structure.animal_pos ;
%  traking_time =        traking_structure.time          ;


tic



npoints = 1;
d = datestr(datetime('now'));
d = strrep(d, '-', '_');
d = strrep(d, ' ', '_');
d = strrep(d, ':', '_');
%% if already started load last traking structre values
% vidObj              = VideoReader(video_loc);
% 
% frames2stract = traking_structure.frames2stract;
% traking_values_animal = traking_structure.animal_pos;
% traking_time = traking_structure.time ;
% 
% % ranagetrak = [465 920];
% % video_loc           = '_NMMTT_230929 23-10-09 15-00-35.avi';
% 
% frame_rate          = 30;
% starting_traking    = frames2stract(1); %up to 605
% n_frames2track      = numel(frames2stract);
% 
% first_nan           = current_min(find(isnan(traking_values_animal)));
%%
startin_point = 1;
brillance_val = 50;
fig = figure('units','normalized','outerposition',[.25 .25 .5 .75]);
for fn= startin_point :numel(frames2stract)

    if mod(fn,200)==0
        close(fig)

        traking_structure   = [];

        traking_structure.frames2stract = frames2stract;
        traking_structure.animal_pos    = traking_values_animal;
        traking_structure.time          = traking_time;

        d = datestr(datetime('now'));
        d = strrep(d, '-', '_');
        d = strrep(d, ' ', '_');
        d = strrep(d, ':', '_');
        save(strcat('traking_structure_',d ), 'traking_structure');
        fig = figure('units','normalized','outerposition',[.25 .25 .5 .75]);

    end

    current_frame = frames2stract(fn);
    current_image = read(vidObj,current_frame );
    hold off
    imagesc(current_image+brillance_val)
    hold on
    title(['Frame#', num2str(current_frame), ' Time =', num2str(current_frame/frame_rate), ' ( fn= ' ,num2str(fn), ' from ', num2str(numel(frames2stract)), ' / ', num2str(100*fn/n_frames2track), '%)'])
    plot(traking_values_animal(fn,1), traking_values_animal(fn,2), 'xg', 'MarkerSize',10)
    new_traking_point = ginput(1);
    traking_values_animal(fn,:) = new_traking_point;
    plot(new_traking_point(1), new_traking_point(2), '.r', 'MarkerSize',10)
end


toc
traking_structure.frames2stract = frames2stract;
traking_structure.animal_pos    = traking_values_animal;
traking_structure.time          = traking_time;


        save(strcat('traking_structure_',d ), 'traking_structure');
%% in case you want to save with a specific name

  save('traking_structure PD2', 'traking_structure');


%% traking second animal




% frames2stract       = traking_structure.frames2stract;
vidObj              = VideoReader(video_loc);
n_frames2track = numel(frames2stract);

traking_values_partner      = nan(n_frames2track,2);
traking_values_partner(1:size(traking_values_animal,1),:) = traking_values_animal;
%% actually traking
brillance_val = 50;
tic
reset_figure = 0;
% frames2correct=1;
% starting_frame =frames2correct(1);
starting_frame = 1;
fig  = figure('units','normalized','outerposition',[0 .25 .5 .75]);
% fig.Position = p(1,:);
% f.WindowState = "maximized"
traking_structure.partner_pos = traking_values_partner;
for fn= starting_frame:numel(frames2stract)

    if mod(fn,200)==0
       close(fig)
       fig  = figure('units','normalized','outerposition',[.25 .25 .5 .75]);
       traking_structure.partner_pos = traking_values_partner;


        d = datestr(datetime('now'));
        d = strrep(d, '-', '_');
        d = strrep(d, ' ', '_');
        d = strrep(d, ':', '_');
        save(strcat('traking_structure_P_',d ), 'traking_structure');
    end
    hold off
    current_frame = frames2stract(fn);
    current_image = read(vidObj,current_frame );

    imagesc(current_image+brillance_val)
    hold on
    title(['Frame#', num2str(current_frame), ' Time =', num2str(current_frame/frame_rate), ' ( fn= ' ,num2str(fn), ' / ', num2str(100*fn/n_frames2track), '%)'])
    plot(traking_structure.animal_pos(fn,1), traking_structure.animal_pos(fn,2), 'xg', 'MarkerSize',10)
    plot(traking_structure.partner_pos(fn,1), traking_structure.partner_pos(fn,2), 'g.', 'MarkerSize',20)
    new_traking_point = ginput(1);
    traking_values_partner(fn,:) = new_traking_point;
    hold on
    plot(new_traking_point(1), new_traking_point(2), '.r', 'MarkerSize',10)
    % pause(.1)
end

       traking_structure.partner_pos = traking_values_partner;
toc


%% correct animal traking values

% frames2correct = [297,301,313,324,356:364,366:367,378:379,391,393,429,433,447,466,469,471,478,480,481,482,541,542,548,586,598]
% frames2correct = [297,301,313,324,356:364,366:367,378:379,391,393,429,433,447,466,469,471,478,480,481,482,541,542,548,586,598,602,638,661,668,669,673:677,680,686,702,703,731,738,739,743:746,780,837,858,860,949,951,974]
frames2correct = 3:20;
 new_Traking_Values = nan(numel(frames2correct),2);
fig  = figure('units','normalized','outerposition',[0 0 1 1]);
frame_count = 1;
for fn= frames2correct

    hold off
    current_frame = frames2stract(fn);
    current_image = read(vidObj,current_frame );
    imagesc(current_image+brillance_val)
    hold on
    title(['Frame#', num2str(current_frame), ' Time =', num2str(current_frame/frame_rate), ' ( fn= ' ,num2str(fn), ' / ', num2str(100*fn/n_frames2track), '%)'])
    plot(traking_structure.animal_pos(fn,1), traking_structure.animal_pos(fn,2), 'rx', 'MarkerSize',20)
    plot(traking_structure.partner_pos(fn,1), traking_structure.partner_pos(fn,2), 'g.', 'MarkerSize',20)

    new_traking_point = ginput(1);
    if new_traking_point(1)<=721
        new_Traking_Values(frame_count,:) = new_traking_point;
    else
        new_Traking_Values(frame_count,:) = traking_structure.animal_pos(fn,:);
    end
    hold on
    plot(new_traking_point(1), new_traking_point(2), '.r', 'MarkerSize',16)
    frame_count = frame_count+1;
    traking_structure.animal_pos(frames2correct,:) = new_Traking_Values;
end


%%
traking_structure.animal_pos = traking_values_animal;
traking_structure.partner_pos = traking_values_partner;
save('traking_structure_P','traking_structure')

%% define area borders
figure
current_frame = traking_structure.frames2stract(end)-100;
current_image = read(vidObj,current_frame );
imagesc(current_image+25)


outher_border =  drawpolygon;

inner_border    =  drawpolygon;
inner_box =  drawpolygon;
%%
o_border    = outher_border.Position ;
i_border    = inner_border.Position;
i_box       = inner_box.Position;

traking_structure.outher_border = o_border;
traking_structure.inner_border = i_border;
traking_structure.inner_box = i_box;


%%
% 
% traking_structure.outher_border = o_border;
% traking_structure.inner_border = i_border;
% traking_structure.inner_box = i_box;

[frames2stract,sorted_frames] = sort(traking_structure.frames2stract);

repeated = find(diff(frames2stract)==0);
frames2stract(repeated)= [];
smoothed_animal = traking_structure.animal_pos(sorted_frames,:);
smoothed_animal(repeated,:) = [];
all_frames = traking_structure.frames2stract(1):traking_structure.frames2stract(end);
all_frames_pos_animal = [interp1(frames2stract,smoothed_animal(:,1),all_frames, 'cubic' ); ...
                        interp1(frames2stract,smoothed_animal(:,2),all_frames, 'cubic' )]';

smoothed_animal = all_frames_pos_animal;
smoothed_animal(:,1) =  smoothdata(all_frames_pos_animal(:,1), 3,'loess');
smoothed_animal(:,2) =  smoothdata(all_frames_pos_animal(:,2), 3,'loess');

smoothed_partner = traking_structure.partner_pos(sorted_frames,:);
smoothed_partner(repeated,:) = [];
all_frames_pos_partner = [interp1(frames2stract,smoothed_partner(:,1),all_frames, 'cubic' ); ...
                        interp1(frames2stract,smoothed_partner(:,2),all_frames, 'cubic' )]';
smoothed_partner = all_frames_pos_partner;
smoothed_partner(:,1) =  smoothdata(all_frames_pos_partner(:,1), 3,'loess');
smoothed_partner(:,2) =  smoothdata(all_frames_pos_partner(:,2), 3,'loess');

%%
figure
new_start = 60;



%% last control before savin5
for j=new_start:numel(all_frames)
new_start = j;
    current_min         = max(1, j-9);

    xy_animal   =smoothed_animal(current_min:j,:);

    xy_partner  = smoothed_partner(current_min:j,:);
    hold off
    current_frame = all_frames(j);
    current_image = read(vidObj,current_frame );
        imagesc(current_image)
        hold on
    % plot([i_border(:,1);i_border(1,1)], [i_border(:,2);i_border(1,2)], 'w')
    % plot([i_box(:,1);i_box(1,1)], [i_box(:,2);i_box(1,2)], ':w')


    plot(xy_animal(:,1), xy_animal(:,2), 'r')
    plot(xy_animal(end,1), xy_animal(end,2), '.r', 'MarkerSize',10)
    plot(xy_partner(:,1), xy_partner(:,2), 'g')
    plot(xy_partner(end,1), xy_partner(end,2), '.g', 'MarkerSize',10)
    title(num2str([all_frames(j) all_frames(end)]), 'FontSize', 14)
    pause(0.01)
end


%%
[frames2stract,sorted_frames] = sort(traking_structure.frames2stract);

repeated = find(diff(frames2stract)==0);
frames2stract(repeated)= [];
animal_pos = traking_structure.animal_pos(sorted_frames,:);
animal_pos(repeated,:) = [];
partner_pos = traking_structure.partner_pos(sorted_frames,:);
partner_pos(repeated,:) = [];
traking_structure.partner_pos = partner_pos;
traking_structure.frames2stract = frames2stract;
traking_structure.animal_pos = animal_pos;
traking_structure.time = traking_structure.frames2stract/30;


save('traking_structure_C2_P', 'traking_structure')