function norm_reg = l2_normalize_regressors(reg)
%L2_NORMALIZE_REGRESSORS L2-normalizes each row of a regressor matrix
%   norm_reg = l2_normalize_regressors(reg)
%   - reg: [n_bouts x n_bins] matrix with NaNs allowed
%   - norm_reg: same size, each row normalized to unit L2 norm (ignoring NaNs)

    norm_reg = reg;
    for i = 1:size(reg,1)
        row = reg(i,:);
        valid = ~isnan(row);
        if any(valid)
            norm_val = norm(row(valid),2);
            if norm_val > 0
                norm_reg(i,valid) = row(valid) / norm_val;
            end
        end
    end
end