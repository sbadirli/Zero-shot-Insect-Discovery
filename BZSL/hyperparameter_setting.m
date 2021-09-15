% This function parses the input variables into parameters
function [k_0,k_1, m, mu_0, s, scatter, num_iter, pca_dim, tuning] = hyperparameter_setting(hy)
    
    p               = inputParser;      
    dim             = 500;
    % Default values for hyperparameters
    default_k0      = 0.01; 
    default_k1      = 10;
    default_m       = 5*dim; 
    default_mu0     = 0; 
    default_s       = 1; 
    default_iter    = 1;
    default_pca     = 0;
    default_scatter = 0;
    default_tuning  = false;
     
    addOptional(p,'kappa_0',default_k0,@isnumeric);
    addOptional(p,'kappa_1',default_k1,@isnumeric);
    addOptional(p,'cov_shape',default_m,@isnumeric);
    addOptional(p,'prior_mean',default_mu0,@isnumeric);
    addOptional(p,'prior_covscale',default_s,@isnumeric);
    addOptional(p,'iter',default_iter,@isnumeric);
    addOptional(p,'pca', default_pca,@isnumeric);
    addOptional(p, 'scatter', default_scatter,@isnumeric);
    addOptional(p, 'tuning', default_tuning, @islogical)
    
    parse(p, hy{:});

    k_0      = p.Results.kappa_0;
    k_1      = p.Results.kappa_1;
    m        = p.Results.cov_shape;
    num_iter = p.Results.iter;
    pca_dim  = p.Results.pca;
    mu_0     = p.Results.prior_mean;
    s        = p.Results.prior_covscale;
    scatter  = p.Results.scatter;
    tuning   = p.Results.tuning;
    
    if p.Results.prior_mean == 0
        mu_0 = zeros(1, dim);
    elseif p.Results.scatter == 0
        scatter  = eye(dim);
    end
    
        
