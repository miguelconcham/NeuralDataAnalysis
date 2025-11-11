function [population_mean, population_pctl, excess_activation, excess_inhibition] = estiamte_activation(all_neurons_response,session,onset_time, response_window, n_rand, pctl2plot)

response_index = mean(all_neurons_response(:,onset_time>response_window(1) & onset_time<response_window(2)),2);
session_list = unique(session);
stacked_activated = nan(numel(session_list), numel(onset_time));
for sep = 1:numel(session_list)
    this_session_activated =  all_neurons_response(session==session_list(sep) & response_index>=0,:);

    if size(this_session_activated,1)==1
        stacked_activated(sep,:) = this_session_activated;
    elseif size(this_session_activated,1)>1
        stacked_activated(sep,:) = mean(this_session_activated);
    else
        stacked_activated(sep,:) = nan(1, numel(onset_time));
    end
end

real_mean_activation = mean(stacked_activated,'omitmissing');
population_mean(1,:) = real_mean_activation;


stacked_inhibited = nan(numel(session_list), numel(onset_time));
for sep = 1:numel(session_list)
    this_session_inhibited =  all_neurons_response(session==session_list(sep) & response_index<0,:);

    if size(this_session_inhibited,1)==1
        stacked_inhibited(sep,:) = this_session_inhibited;
    elseif size(this_session_inhibited,1)>1
        stacked_inhibited(sep,:) = mean(this_session_inhibited);
    else
        stacked_inhibited(sep,:) = nan(1, numel(onset_time));
    end
end

real_mean_inhibition = mean(stacked_inhibited,'omitmissing');
population_mean(2,:) = real_mean_inhibition;



resampling_matrices = repmat(1:size(all_neurons_response,2),size(all_neurons_response,1),1);
resampling_matrices = repmat(resampling_matrices, 1 ,1,n_rand);
resampling_matrices =  permute(resampling_matrices, [3 1 2]);

for j=1:size(all_neurons_response,1)
    last_no_nan    = max(find(~isnan(all_neurons_response(j,:))));
    for n=1:n_rand
    resampling_matrices(n,j,1:last_no_nan) =  mod((1:last_no_nan) + round(rand*last_no_nan),last_no_nan)+1;
    end
end

positive_mean_distribution = nan(n_rand, numel(onset_time));
negative_mean_distribution = nan(n_rand, numel(onset_time));
for n=1:n_rand
    stacked_activated = nan(numel(session_list), numel(onset_time));
    stacked_inhibited = nan(numel(session_list), numel(onset_time));

    shufled_all_neurons_response = nan(size(all_neurons_response));
    for j=1:size(all_neurons_response,1)
        shufled_all_neurons_response(j,:) = all_neurons_response(j,squeeze(resampling_matrices(n,j,:)));
    end
    response_index = mean(shufled_all_neurons_response(:,onset_time>response_window(1) & onset_time<response_window(2)),2);

    for sep = 1:numel(session_list)
        this_session_activated =  shufled_all_neurons_response(session==session_list(sep) & response_index>=0,:);

        if size(this_session_activated,1)==1
            stacked_activated(sep,:) = this_session_activated;
        elseif size(this_session_activated,1)>1
            stacked_activated(sep,:) = mean(this_session_activated);
        else
            stacked_activated(sep,:) = nan(1, numel(onset_time));
        end

        this_session_inhibited =  shufled_all_neurons_response(session==session_list(sep) & response_index<0,:);

        if size(this_session_inhibited,1)==1
            stacked_inhibited(sep,:) = this_session_inhibited;
        elseif size(this_session_inhibited,1)>1
            stacked_inhibited(sep,:) = mean(this_session_inhibited);
        else
            stacked_inhibited(sep,:) = nan(1, numel(onset_time));
        end

    end
    positive_mean_distribution(n,:) = mean(stacked_activated,'omitmissing');
    negative_mean_distribution(n,:) = mean(stacked_inhibited,'omitmissing');
end


sorted_values = sort(positive_mean_distribution);
[~, loc] = min(abs(sorted_values-repmat(real_mean_activation,n_rand,1)));
population_pctl(1,:) = (n_rand-loc)/n_rand;


excess_activation(1,:) = sorted_values(round(pctl2plot*n_rand/100),:);
excess_inhibition(1,:) = sorted_values(round((100-pctl2plot)*n_rand/100),:);

sorted_values = sort(negative_mean_distribution);
[~, loc] = min(abs(negative_mean_distribution-repmat(real_mean_inhibition,n_rand,1)));
population_pctl(2,:) = (loc)/n_rand;

excess_activation(2,:) = sorted_values(round(pctl2plot*n_rand/100),:);
excess_inhibition(2,:) = sorted_values(round((100-pctl2plot)*n_rand/100),:);




