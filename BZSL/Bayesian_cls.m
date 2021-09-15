    % Main function for running the algorithm
    % Inputs: (Musts) 
    %   training data:              x_tr, y_tr
    %   Test data (seen):           x_ts_s, y_ts_s    
    %   Test data (unseen):         x_ts_us, y_ts_us 
    %   Genus labels:               G                            
    % Optional inputs:
    %   Hyperparameters:            ['kappa_0','kappa_1','prior_mean','prior_covscale']
    %                               'prior_mean','prior_covscale' as mu_0 & s
    %                               'cov_shape' stands for m in the paper
    %                               
    %   Number of iters:            'iter', accepts integer
    %   # components for PCA :      'pca',  accepts positive integer. 
    %                               0 means no need for PCA.
    %   Tuning option:              'tuning', true or false.
    % Outputs:
    %   Accuracy for seen classes:   seen_acc
    %   Accuracy for unseen classes: unseen_acc  
    %   Harmonic mean:               H
    
    

function [seen_acc, unseen_acc, H, acc_per_sclass, acc_per_usclass, prob_mat_s, prob_mat_us, class_id] = Bayesian_cls(x_tr, y_tr, x_ts_us, y_ts_us, x_ts_s, y_ts_s, G, varargin)
    % Seen and unseen classes
    tic
    d0          = size(x_tr, 2);

    % Parsing passed parameters and  hyperparameters from tuning
    params      = varargin;
    [k_0,k_1, m, mu_0, s, scatter, num_iter, pca_dim, tuning] = hyperparameter_setting(params);
    
    % num_iter for repeating the procedure several times to eleminate
    % randomness
    % You may change the # features to use by changing d0
    if pca_dim 
        % Dimentionality reduction from PCA
        C       = cov(x_tr);
        [vv, ~] = eig(C);
        x_tr    = x_tr*vv(:,end-pca_dim+1:end);
        x_ts_s  = x_ts_s*vv(:,end-pca_dim+1:end);
        x_ts_us = x_ts_us*vv(:,end-pca_dim+1:end);
        
        d0      = pca_dim;
    end
        
    % Mixing feature positions
    if tuning
        for i=1:num_iter
        fin{i}  = 1:d0;
        end
    else 
        for i=1:num_iter
            tmp     = randperm(d0);
            fin{i}  = tmp(1:d0);
            fin{i}  = 1:d0;
        end
    end
    
        

    % Main for loop for the calculations
    for iter=1:numel(fin)
        
        % training data
        xn         = x_tr(:,fin{iter});
        yn         = y_tr;
        % Test data from seen and unseen classes (GZSL)
        xt_unseen  = x_ts_us(:,fin{iter});
        xt_seen    = x_ts_s(:,fin{iter});
        
        % Pre-calculation of Psi (prior covariance) from tuned scale s, and
        % scatter. The reason behind this if statement is that we dont want
        % repeat this precalculation in hypertuning since it is expensive
        % in time but we want to calculate this values with new data during
        % testing
        if tuning
            Psi=(m-d0-1)*scatter/s;
        else
            [mu_0, scatter] = calculate_priors(xn, yn);
            Psi=(m-d0-1)*scatter/s;
        end
           
        % Class predictive cov, mean and DoF 
        [Sig_s,mu_s,v_s,class_id,Sigmas]    = ppd_derivation(xn,yn,G, Psi,mu_0,m,k_0,k_1);
        fprintf('Model training time: \n')
        toc
        % Inference phase
        tic
        [ypred_unseen(:,iter), prob_mat_us] = predict(xt_unseen, Sig_s, mu_s, v_s, class_id);
        [ypred_seen(:,iter), prob_mat_s]    = predict(xt_seen, Sig_s, mu_s, v_s, class_id);
        fprintf('Model inference time: \n')
        toc


    end
      
    %%% Performance calculation %%%
    
    % Mode of iterations to alleviate the effect of r.v.
    ypred         = mode(ypred_unseen,2);
    [acc_per_usclass, unseen_acc] = evaluate(y_ts_us, ypred, G, 'unseen'); % Unseen classes
    % Accuracy calculation for seen classes
    ypred          = mode(ypred_seen,2);
    [acc_per_sclass, seen_acc] = evaluate(y_ts_s, ypred, G, 'seen');       % Seen classes


    % Harmonic mean for seen and unseen classes acc. from Y. Xian paper
    H = 2*unseen_acc*seen_acc/(unseen_acc + seen_acc);   

end