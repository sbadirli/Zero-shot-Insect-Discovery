% This function calculates the prior mean and prior covarince in advance to
% feed in Bayesian classifier
function [mu_0, scatter] = calculate_priors(Data, labels)
     
    [~, dim]    = size(Data);
    uy          = unique(labels);
    nc          = length(uy);

    scatters    = zeros(dim,dim,nc);
    class_means = zeros(nc,dim);
    for j=1:length(uy)
        scatters(:,:,j)  = cov(Data(labels==uy(j),:));
        class_means(j,:) = mean(Data(labels==uy(j),:),1);
    end  
    scatter     = mean(scatters,3);
    mu_0        = mean(class_means,1);
end