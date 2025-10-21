function array_shifted = shift_rows_conditional(array, shifts, time, length_val)
% array:      n x T
% shifts:     1 x n (or n x 1)
% time:       1 x T
% length_val: 1 x n (or n x 1)

[n, T] = size(array);
array_shifted = array; % initialize output

for i = 1:n
    mask = time < length_val(i);       % logical mask for this row
    vals = array(i, mask);             % elements to shift
    k = shifts(i);
    if ~isempty(vals)
        vals = circshift(vals, [0, k]); % circular shift only masked elements
        array_shifted(i, mask) = vals;  % put back into row
    end
end
end
