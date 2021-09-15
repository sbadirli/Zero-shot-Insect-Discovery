 
function [k0, k1, mm, a0, b0, mu_0, s] = hyperparameter_tuning(xtrain,ytrain,xtest_unseen,ytest_unseen,xtest_seen,ytest_seen, G, varargin)
    
    if nargin<2
        dim = size(xtrain, 2);
    else
        dim  = varargin{2};
    end
   
    % Tuning range for the parameters
    k0_range = [0.1 10];
    k1_range = [1 10];
    s_range  = [0.5 1 5];
    

    bestH    = 0; best_acc_s = 0; best_acc_us = 0;
    
        
    % PCA for dimentionality reduction
    fprintf('Applying PCA to reduce the dimension...\n')
    C            = cov(xtrain);
    [vv, ~]      = eig(C);
    xtrain       = xtrain*vv(:,end-dim+1:end);
    xtest_seen   = xtest_seen*vv(:,end-dim+1:end);
    xtest_unseen = xtest_unseen*vv(:,end-dim+1:end);



    % Precalculation of class means and scatter matrices for unconstrained model
    [mu_0, scatter] = calculate_priors(xtrain, ytrain);

    % m range is determined by the dim of data
    m_range  = [2*dim 5*dim 10*dim 25*dim 100*dim];
    fprintf('Tuning is getting started...\n')
    for k_0=k0_range
        for k_1=k1_range
            for m=m_range
                for ss=s_range
                    %Psi=(m-dim-1)*scatter/s;
                    tic
                    [seen_acc, unseen_acc, H, s_cls_acc, us_cls_acc, pb_s, pb_us, class_id] = Bayesian_cls(xtrain,ytrain,xtest_unseen,ytest_unseen,xtest_seen,ytest_seen, G,'tuning',true,...
                                        'kappa_0', k_0, 'kappa_1', k_1, 'cov_shape', m, 'prior_mean', mu_0,'prior_covscale', ss,'scatter', scatter, 'pca', 0);

                    % Print out when there is an improvement in Harmonic mean
                    if H>bestH  
                        bestH = H;% best_acc_s = acc_s; best_acc_us = acc_us;
                        k0 = k_0; k1 = k_1; mm = m; s  = ss; K = kk;
                    end
                    toc

                    disp(['Unseen classes mean accuracy=' num2str(unseen_acc) ]);
                    disp(['Seen classes mean accuracy=' num2str(seen_acc) ]);
                    disp(['GZSL: H=' num2str(H)]);
                    disp(['k0=' num2str(k_0)]);
                    disp(['k1=' num2str(k_1)]);
                    disp(['m=' num2str(m)]);
                    disp(['s=' num2str(ss)]);


                end
            end
        end
    end

end