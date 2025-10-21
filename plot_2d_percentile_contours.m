function plot_2d_percentile_contours(data, percentiles, ax, color)
    % data: m×2 array of points
    % percentiles: vector of percentiles to plot (e.g., [50, 75, 90])
    % ax (optional): axis handle to plot into

    if nargin < 3 || isempty(ax)
        figure;
        ax = gca;
    end

    % Create grid for KDE
    x = linspace(min(data(:,1)), max(data(:,1)), 200);
    y = linspace(min(data(:,2)), max(data(:,2)), 200);
    [X, Y] = meshgrid(x, y);
    grid_points = [X(:), Y(:)];

    % Kernel density estimation
    [f, ~] = ksdensity(data, grid_points);
    Z = reshape(f, size(X));

    % Normalize and compute cumulative density
    sorted_Z = sort(Z(:), 'descend');
    cum_density = cumsum(sorted_Z);
    cum_density = cum_density / max(cum_density);

    % Find density thresholds for desired percentiles
    contour_levels = zeros(size(percentiles));
    for i = 1:numel(percentiles)
        p = percentiles(i) / 100;
        idx = find(cum_density >= p, 1, 'first');
        contour_levels(i) = sorted_Z(idx);
    end

    % Plot into given axis
    axes(ax); hold(ax, 'on');
    contour(ax, X, Y, Z, contour_levels, 'ShowText', 'off', 'EdgeColor',color, 'HandleVisibility','off');
    % scatter(ax, data(:,1), data(:,2), 5, 'k.');
    % title(ax, '2D Distribution Percentile Contours');
    % xlabel(ax, 'Dimension 1'); ylabel(ax, 'Dimension 2');
end