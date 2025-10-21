function [f,meanPxx,ci] = avg_spectrum_CI_pmtm(x,Fs,freqRange,winLength,overlap,nboot,timeBW)
    % Multitaper PSD with bootstrap CI
    % x: signal
    % Fs: sampling frequency
    % freqRange: [fmin fmax]
    % winLength: window length in samples
    % overlap: overlap in samples
    % nboot: number of bootstrap resamples
    % timeBW: time-bandwidth product for pmtm (typ. 2-4 for LFP)

    % Segment signal
    step = winLength - overlap;
    nSegments = floor((length(x)-overlap)/step);
    segments = zeros(winLength,nSegments);
    for i = 1:nSegments
        idx = (1:winLength) + (i-1)*step;
        segments(:,i) = x(idx);
    end

    % Compute multitaper PSD for each segment
    for i = 1:nSegments
        [pxx,f_full] = pmtm(segments(:,i), timeBW, [], Fs);
        if i==1
            allPxx = zeros(length(pxx),nSegments);
        end
        allPxx(:,i) = pxx;
    end

    % Select frequency range
    idx = f_full >= freqRange(1) & f_full <= freqRange(2);
    f = f_full(idx);
    allPxx = allPxx(idx,:);

    % Mean PSD
    meanPxx = mean(allPxx,2);

    % Bootstrap CI (resample across segments)
    bootPxx = bootstrp(nboot,@(s) mean(allPxx(:,s),2)',1:nSegments);
    ci = prctile(bootPxx,[2.5 97.5])';  % 95% CI
end
