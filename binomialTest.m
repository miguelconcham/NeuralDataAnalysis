function pval = binomialTest(k, n, p0)
    % Binomial test for H0: p = p0
    % k = observed successes
    % n = number of trials
    % p0 = null probability

    % lower tail
    p_left  = binocdf(k, n, p0);

    % upper tail
    if k > 0
        p_right = 1 - binocdf(k-1, n, p0);
    else
        p_right = 1;  % special case: k=0
    end

    % two-sided p-value
    pval = 2 * min(p_left, p_right);
    if pval > 1
        pval = 1;
    end
end
