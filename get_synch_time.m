function [NPX_start_time] = get_synch_time(NPX, probe_name)
NPX_start_time = NaN;
fid = fopen('sync_messages.txt');

not_imported = true;
tline = fgetl(fid);
while not_imported && ischar(tline)
    
    
    
    content             = strsplit(tline,' ');
    if ismember('subProcessor:', content) 
        if str2double(content{find(ismember(content,'subProcessor:'))+1})==NPX
            
            start_time_string   = content{end};
            start_time_string   = strsplit(start_time_string, '@');
            NPX_start_time      = str2double(start_time_string{1});
            not_imported= false;
        end
    elseif  ismember(probe_name,content)
            start_time_string   = content{end};
            NPX_start_time      = str2double(start_time_string);
            not_imported= false;
    end
    tline = fgetl(fid);
end

if ~ischar(tline) && not_imported
    disp(['SYNCH TIME NOT FOUND FOR NPX ' num2str(NPX)])
else
    disp(['SYNCH TIME = ' num2str(NPX_start_time)])
end