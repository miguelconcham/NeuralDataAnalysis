function mi = image_mutual_info(I1, I2, nbins)
    % Convert to double
    I1 = double(I1(:));
    I2 = double(I2(:));

    % Define bins
    edges1 = linspace(min(I1), max(I1), nbins+1);
    edges2 = linspace(min(I2), max(I2), nbins+1);

    % Discretize intensities
    X = discretize(I1, edges1);
    Y = discretize(I2, edges2);

    % Remove any NaNs (shouldn't happen if edges cover min/max)
    valid = ~isnan(X) & ~isnan(Y);
    X = X(valid);
    Y = Y(valid);

    % Joint histogram
    jointHist = accumarray([X Y], 1, [nbins nbins]);
    jointProb = jointHist / sum(jointHist(:));

    % Marginal probabilities
    pX = sum(jointProb, 2);
    pY = sum(jointProb, 1);

    % Compute MI
    mi = 0;
    for i = 1:nbins
        for j = 1:nbins
            if jointProb(i,j) > 0
                mi = mi + jointProb(i,j) * log2(jointProb(i,j)/(pX(i)*pY(j)));
            end
        end
    end
end



