save_NPY = true;
sample_Rate = 30000;
sample_length_sec = 60;
LFP_sample_rate =2500;
file_pointer    = fopen('continuous.dat', 'r');
con_fil = dir('continuous.dat');
length_file = con_fil.bytes/(384*2);
LFP_length =ceil(length_file*LFP_sample_rate/sample_Rate);
new_LFP = zeros(384,LFP_length, 'int16');
frewind(file_pointer)

iteration =0;
LFP = fread(file_pointer,[384 sample_Rate*sample_length_sec],'int16');
tic
while size(LFP,2)==sample_Rate*sample_length_sec
        current_indexes =(iteration*sample_length_sec*LFP_sample_rate + 1):((iteration+1)*sample_length_sec*LFP_sample_rate);
    toc
    disp(['LOADING ' , num2str(current_indexes(1)/LFP_sample_rate) , ' s to ', num2str(current_indexes(end)/LFP_sample_rate) , 's from ', num2str(double(length_file)/sample_Rate), ' s'])
    for j=1:384
        new_LFP(j  ,current_indexes) = downsample(LFP(j,:), sample_Rate/LFP_sample_rate);
    end
    LFP = fread(file_pointer,[384 sample_Rate*sample_length_sec],'int16');
    iteration =iteration+1;
end


remaining_aux = downsample(LFP(1,:), sample_Rate/LFP_sample_rate);
disp(['LOADING LAST ', num2str(numel(remaining_aux)/2500) , ' s '])
remaining_indexes = (iteration*sample_length_sec*LFP_sample_rate + 1):(iteration*sample_length_sec*LFP_sample_rate + numel(remaining_aux));
missing_LFP_samples = remaining_indexes(end)-size(new_LFP,2);

if missing_LFP_samples>0
    new_LFP = [new_LFP,zeros(384,missing_LFP_samples, 'int16')];
elseif missing_LFP_samples<0
    new_LFP(:, (end+missing_LFP_samples+1):end) = [];
end

for j=1:384
    new_LFP(j  ,remaining_indexes) = downsample(LFP(j,:), sample_Rate/LFP_sample_rate);
end

LFP = new_LFP;
clear new_LFP
disp('Saving')
save('LFP','LFP', '-v7.3')
%%
if save_NPY
    for j=1:384
        current_ch = LFP(j,:);
        disp(['Saving CH', num2str(j), '.npy'])
        writeNPY(current_ch,[ 'CH', num2str(j), '.npy']);
    end
end

