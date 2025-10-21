function [mean_value, ci] = plot_ci(time,matrix2plot,inputArsubplot_axes2, color, plot_bool)
face_alpha = .25;
line_width = 1.5;
mean_value = [];
ci         = [];
if size(matrix2plot,1)==1
    mean_value = matrix2plot;
    ci          = nan(2,size(matrix2plot,2));
else
    mean_value = mean(matrix2plot, "omitmissing");
     ci          = nan(2,size(matrix2plot,2));

     for t=1:size(matrix2plot,2)
          no_nan = ~isnan(matrix2plot(:,t));
          if sum(no_nan)>2
              ci([1 2],t) = mean_value(t) + [-std(matrix2plot(no_nan,t)) std(matrix2plot(no_nan,t))]/sqrt(sum(no_nan));
          end
     end    
end

    axes(inputArsubplot_axes2)
    if plot_bool
    fill([time fliplr(time)],[ ci(1,:) fliplr( ci(2,:))], color, 'FaceAlpha',face_alpha, 'EdgeColor','none', 'HandleVisibility','off')
    end
    hold on
    plot(time,mean_value, color, 'LineWidth',line_width)

