function [f,meanPxx,ci] = avg_spectrum_CI(x,Fs,freqRange,winLength,overlap,nboot)
    % x: signal
    % Fs: sampling frequency
    % freqRange: [fmin fmax]
    % winLength: window length in samples
    % overlap: overlap in samples
    % nboot: number of bootstrap resamples for CI

    % Segment signal
    step = winLength - overlap;
    nSegments = floor((length(x)-overlap)/step);
    segments = zeros(winLength,nSegments);
    for i = 1:nSegments
        idx = (1:winLength) + (i-1)*step;
        segments(:,i) = x(idx);
    end

    % Compute PSD for each segment
    for i = 1:nSegments
        [pxx,f] = pwelch(segments(:,i), [], [], [], Fs);
        if i==1
            allPxx = zeros(length(pxx),nSegments);
        end
        allPxx(:,i) = pxx;
    end

    % Select frequency range
    idx = f >= freqRange(1) & f <= freqRange(2);
    f = f(idx);
    allPxx = allPxx(idx,:);

    % Mean PSD
    meanPxx = mean(allPxx,2);

    % Bootstrap CI
    bootPxx = bootstrp(nboot,@(s) mean(allPxx(:,s),2)',1:nSegments);  % resample across segments
    ci = prctile(bootPxx,[2.5 97.5],1); % 95% CI
    ci = ci'; % transpose to match f

end