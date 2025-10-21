function extractStructFields(s)
% Extract fields from a struct into the caller workspace
% Usage: extractStructFields(myStruct)

    if ~isstruct(s)
        error('Input must be a structure.');
    end

    fields = fieldnames(s);

    for i = 1:numel(fields)
        fieldName = fields{i};
        fieldValue = s.(fieldName);
        assignin('caller', fieldName, fieldValue);
    end
end