function [mode_angle, edges, counts, counts_smooth] = circ_mode_chat(angles, nbins, varargin)
% CIRC_MODE  Circular-mode (peak) from smoothed polar counts.
%   [mode_angle, edges, counts, counts_smooth] = circ_mode(angles, nbins, ...)
%
% Inputs
%   angles : vector of angles (radians), any range is fine
%   nbins  : number of histogram bins (e.g., 36 -> ~10°)
%
% Name-Value options
%   'SigmaBins' : Gaussian smoothing width in *bins* (default 1.5)
%   'Plot'      : true/false to plot polar histogram with peak (default false)
%
% Outputs
%   mode_angle   : location of peak (radians, wrapped to [-pi, pi])
%   edges        : histogram edges (radians)
%   counts       : raw counts per bin
%   counts_smooth: smoothed counts (circularly smoothed)

  p = inputParser;
  addParameter(p, 'SigmaBins', 1.5, @(x) isnumeric(x) && isscalar(x) && x>0);
  addParameter(p, 'Plot', false, @(x) islogical(x) && isscalar(x));
  parse(p, varargin{:});
  sigmaBins = p.Results.SigmaBins;
  doPlot    = p.Results.Plot;

  % Wrap data to [-pi, pi] for consistency
  ang = wrapToPi(angles(:));

  % Histogram on [-pi, pi]
  edges = linspace(-pi, pi, nbins+1);
  counts = histcounts(ang, edges);

  % --- Circular Gaussian smoothing on counts ---
  % Build Gaussian kernel in bin units
  halfw = max(1, ceil(3*sigmaBins));                  % cover ~±3σ
  u = -halfw:halfw;
  g = exp(-0.5*(u/sigmaBins).^2);
  g = g / sum(g);

  % Circularly smooth by tiling counts and convolving, then take center block
  counts_ext = [counts, counts, counts];
  counts_ext_sm = conv(counts_ext, g, 'same');
  counts_smooth = counts_ext_sm(nbins+1 : 2*nbins);

  % Bin centers
  centers = (edges(1:end-1) + edges(2:end)) / 2;

  % Peak from smoothed counts
  [~, imax] = max(counts_smooth);
  mode_angle = wrapToPi(centers(imax));

  % Optional plot
  if doPlot
    figure; 
    polarhistogram(ang, nbins); hold on
    % Overlay peak as a radial line up to the smoothed max height (scaled)
    rmax = max(counts_smooth);
    polarplot([mode_angle mode_angle], [0 rmax], 'LineWidth', 2);
    title(sprintf('Circular mode = %.2f rad (%.1f°)', mode_angle, rad2deg(mode_angle)));
  end
end
