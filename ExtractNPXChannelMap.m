function [chanMap,chanMap0ind,connected,name,shankInd,xcoords,ycoords] = ExtractNPXChannelMap(path_2_settings, name,port_n )
cd(path_2_settings)

fid = fopen('settings.xml');
position = [];

%first find right port
not_found = true;
tline = 0;
while not_found 


    tline = fgetl(fid);
    port = strfind(tline,'port=');
    if ~isempty(port)  & str2double(tline(port+6))==port_n
        disp('Port Found')       
        not_found = false;
    end

    if any(tline==-1)
        disp('port not found, try different port value or check settings file')
    end


end

if ~any(tline==-1)
while isempty(position) 

tline = fgetl(fid);
position = strfind(tline,'<ELECTRODE_XPOS');
end

chanel_Xpos = [];
data2parse = tline(position+16:end);
data2parse = strsplit(data2parse, ' ');

for j=1:numel(data2parse)
    chanel_pos_pair = strsplit(data2parse{j}, '=');

    chanel = str2double(chanel_pos_pair{1}(3:end));
    if ~contains(chanel_pos_pair{2}, '/')
        pos = str2double(chanel_pos_pair{2}(2:end-1));
    else
        pos = str2double(chanel_pos_pair{2}(2:end-3));

    end
    chanel_Xpos = [chanel_Xpos;[chanel pos]];
end


tline = fgetl(fid);


while ~contains(tline, '<')


    data2parse = tline(position+16:end);
    data2parse = strsplit(data2parse, ' ');

    for j=1:numel(data2parse)
        chanel_pos_pair = strsplit(data2parse{j}, '=');

        chanel = str2double(chanel_pos_pair{1}(3:end));
        if isempty(strfind(chanel_pos_pair{2}, '/'))
            pos = str2double(chanel_pos_pair{2}(2:end-1));
        else
            pos = str2double(chanel_pos_pair{2}(2:end-3));

        end      
        chanel_Xpos = [chanel_Xpos;[chanel pos]];
    end
    tline = fgetl(fid);
end

position = strfind(tline,'<ELECTRODE_YPOS ');
chanel_Ypos = [];
data2parse = tline(position+16:end);
data2parse = strsplit(data2parse, ' ');
for j=1:numel(data2parse)
    chanel_pos_pair = strsplit(data2parse{j}, '=');
    chanel = str2double(chanel_pos_pair{1}(3:end));
    if ~contains(chanel_pos_pair{2}, '/')
        pos = str2double(chanel_pos_pair{2}(2:end-1));
    else
        pos = str2double(chanel_pos_pair{2}(2:end-3));

    end
    chanel_Ypos = [chanel_Ypos;[chanel pos]];
end


tline = fgetl(fid);
while ~contains(tline, '<')
    data2parse = tline(position+16:end);
    data2parse = strsplit(data2parse, ' ');
    for j=1:numel(data2parse)
        chanel_pos_pair = strsplit(data2parse{j}, '=');
        chanel = str2double(chanel_pos_pair{1}(3:end));
        if isempty(strfind(chanel_pos_pair{2}, '/'))
            pos = str2double(chanel_pos_pair{2}(2:end-1));
        else
            pos = str2double(chanel_pos_pair{2}(2:end-3));
        end
        chanel_Ypos = [chanel_Ypos;[chanel pos]];
    end
    tline = fgetl(fid);
end





chanMap = chanel_Xpos(:,1)+1;
chanMap0ind = chanel_Xpos(:,1);
connected = true(size(chanMap));
shankInd = ones(size(connected));

xcoords = chanel_Xpos(:,2);
ycoords = chanel_Ypos(:,2);
save(name, 'chanMap','chanMap0ind','connected','name','shankInd','xcoords','ycoords');
else

    chanMap     = NaN;
    chanMap0ind = NaN;
    connected   = NaN;
    name        = NaN;
    shankInd    = NaN;
    xcoords     = NaN;
    ycoords     = NaN;
end

end