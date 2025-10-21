function C = filter_groups_by_mask(A, B)
% FILTER_GROUPS_BY_MASK - Filters consecutive 1-groups in A based on overlap with B.
% 
% Usage:
%   C = filter_groups_by_mask(A, B)
%
% Inputs:
%   A - Logical or binary array (defines groups of consecutive 1s)
%   B - Logical or binary array (mask to check against A's groups)
%
% Output:
%   C - Binary array where only groups from A overlapping with 1s in B are kept

    % Validate input sizes
    if length(A) ~= length(B)
        error('Input arrays A and B must be of the same length.');
    end

    % Find group boundaries in A
    A_diff = diff([0 A 0]);
    starts = find(A_diff == 1);
    ends   = find(A_diff == -1) - 1;

    % Initialize output
    C = zeros(size(A));

    % Process each group
    for i = 1:length(starts)
        group_indices = starts(i):ends(i);
        if any(B(group_indices))
            C(group_indices) = 1;
        end
    end
end
