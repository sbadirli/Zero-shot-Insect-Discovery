% Open set Bayesian Classifier
% Sarkhan Badirli
% PhD candidate, Computer Science
% Purdue University



%%%%%%%%%%%%%%%%   DEMO    %%%%%%%%%%%%%%%%%%%%
% This fuction demostrates the usage of Bayesian classifier on Insect dataset 
% The flow of the Demo is as follows,
% 1.    Load the data with train/val/test splits
%
% 3.    Hyperparameter tuning for the parameters of the classifier. The matlab
% scritp hyperparameter_tuning.m implements this tuning. You can specify the 
% # principal components in the PCA EX: 
% hyperparameter_tuning(Data split..., 'pca', 500);
% The script returns the desired hyperparameters  required for each model.
% For more details about hyperparameters please refer to the Supplementary mat.
%
% 4. After getting the parameters, we used the train and test split
% provided within  datasets. 
%
% 5. Example run:
%     Bayesian_cls(Data split...,'Model','unconstrained', 'tuning', true, 'prior_mean', mu_0, 
%     'prior_cov', psi, 'kappa_0', k_0, 'kappa_1', k_1, 'cov_shape', m, 'pca', 500);

clear all;

%%% Loading the data %%%
datapath = '..\data\';
dataset = 'INSECTS';

fname1=[datapath, dataset, '\data.mat'];   
fname2=[datapath, dataset, '\splits.mat']; 
load(fname1)
load(fname2)
model = 'OSBC_DNA';

%%% Transductive approach %%%
transductive = true;
if transductive
    dd = size(embeddings_dna, 2);
    di = size(embeddings_img, 2);
    rho = 1; 
    
    if strcmp(model, 'OSBC_IMG')
        fprintf('Transductive model works works well if the mapping is from Image to DNA,\n');
        fprintf('Thus we automatically run this version!\n')
    end
    model = 'OSBC_DIT';
    %cv = cvpartition(labels(test_unseen_loc), 'HoldOut', 0.5);
    %st = [trainval_loc, test_unseen_loc(cv.test)];

    st = [trainval_loc, test_unseen_loc, test_seen_loc];

    embeddings_dna = normalize(embeddings_dna, 2, 'zscore');
    embeddings_img = normalize(embeddings_img, 2, 'zscore');

    tic
    V = ridge_regression(embeddings_dna, embeddings_img, st, rho);
    fprintf('Time took for learning map in transductive setup: \n')
    toc
end


% If you want to just reproduce the results from the paper use the following
% hyperparams settings from Supplementary materiall for each model
pca_dim = 500; 
[k_0, k_1, m, s] = load_tuned_params(model);


%%% Model tuning %%%
tuning = false;

if tuning
    % Splitting the training data into training and test data for the use in
    [xtrain, ytrain, xtest_unseen, ytest_unseen, xtest_seen, ytest_seen] = data_split(embeddings_dna, embeddings_img, labels, trainval_loc, ...
                                                                                  train_loc, test_seen_loc, test_unseen_loc, val_seen_loc,...
                                                                                  val_unseen_loc, 'tuning', model);
    if transductive
        %ytest_unseen = G(ytest_unseen);
        x_tr_g = V*features(:, train_loc);

        xtrain = [xtrain; x_tr_g'];
        ytrain = [ytrain; ytrain];
    end
    %Tuning process and optimal parameters from CV
    [k_0, k_1, m, a_0, b_0, mu_0, s, K] = hyperparameter_tuning(xtrain,ytrain,xtest_unseen,ytest_unseen,xtest_seen,ytest_seen,G, 'pca', 500);
end

% Loading training and test splits
[x_tr, y_tr, x_ts_us, y_ts_us, x_ts_s, y_ts_s] = data_split(embeddings_dna, embeddings_img, labels, trainval_loc, train_loc, test_seen_loc,... 
                                                            test_unseen_loc, val_seen_loc, val_unseen_loc, 'test', model);
                                                                            
%%% Data augmentation from Transductive method %%%
if transductive
    x_tr_g = V*embeddings_img(trainval_loc, :)';
    x_tr_g = x_tr_g';
    x_tr = [x_tr; x_tr_g];
    y_tr = [y_tr; y_tr];
end
% PCA for dimensionality reduction
tic
C = cov(x_tr);
[vv, ~] = eig(C);
x_tr    = x_tr*vv(:,end-pca_dim+1:end);
x_ts_s  = x_ts_s*vv(:,end-pca_dim+1:end);
x_ts_us = x_ts_us*vv(:,end-pca_dim+1:end);
pca_dim = 0;
fprintf('Time for PCA: \n')
toc


% Model training and inference
[seen_acc, unseen_acc, H, s_cls_acc, us_cls_acc, pb_s, pb_us, class_id] = Bayesian_cls(x_tr, y_tr, x_ts_us, y_ts_us, x_ts_s, y_ts_s, G, ...
                                               'kappa_0', k_0, 'kappa_1', k_1,'cov_shape', m,'prior_covscale',s, 'pca', pca_dim);

fprintf('Results from k0=%.2f, k1=%.2f, m=%d, s=%.1f\n', k_0, k_1, m, s);
fprintf('Model %s results on %s dataset: Seen acc: %.2f%% Unseen acc: %.2f%%, Harmonic mean: %.2f%% \n', ...
        model, dataset, seen_acc*100, unseen_acc*100, H*100);

