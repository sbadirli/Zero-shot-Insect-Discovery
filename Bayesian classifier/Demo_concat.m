% Sarkhan Badirli
% PhD, Computer Science
% Purdue University



% Bayesian Zero Shot Learning
%%%%%%%%%%%%%%%%   DEMO    %%%%%%%%%%%%%%%%%%%%
% This fuction demostrates the usage of BZSL on the datasets in (Generalized)
% Zero-shot learning domain: CUB, aPY, SUN, AWA & FLO
% The flow of the Demo is as follows,
% 1.    Load the one of the abovementioned data, they have specific train and
% test split nad attached attributes for each class. We have utilized the
% train, test and validation split provided in the datasets itself.
%
% 2.    Partitioning the training data into training and test data for the
% purpose of hyperparamter tuning. tuning_split.m script splits the training
% data into training and test (2 seperate sets: seen and unseen classes).  
% All these datasets have their ZSL specified partitioning and we followed  
% the exact same splitting fashion as in the literature.
%
% 3.    Hyperparameter tuning for the parameters of the BZSL. The matlab
% scritp hyperparameter_tuning.m implements this tuning. The must inputs
% for the tuning is the data split from tuning_split, attriburtes and Model
% version of the BZSL, constrained or unconstrained. If the unconstrained
% model is chosen you can specify the # principal components in the PCA which 
% used as dimensionality reduction technique in unconstrained model. EX: 
%   hyperparameter_tuning(Data split...,'Model', 'unconstrained', 'pca', 500);
% The script returns the desired hyperparameters  required for each model.
% For more details about hyperparameters please refer to the paper.
%
% 4. After getting the parameters, we used the train and test split
% provided within  datasets. We made use of Generalized ZSL setting for the
% split.
%
% 5. Lastly the BZSL model is run. The must inputs for the BZSL is data
% split from step 4, attributes, and the hyperparameters from tuning. Most
% of the hyper-parameters are the same for both models yet there are some
% differences. Please refer to the paper for the full list of
% hyperparameters of each model version. Since random number generation
% involved in the constrained model you could define number of iterations
% in this model to alleviate the randomness on the H score MOreover, there 
% is a tuning option for the BZSL where it is faster in this option if tuning 
% is being performed; ['tuning', true]  Ex: (constrained)
%     Bayesian_ZSL(Data split...,'Model', 'constrained', 'prior_mean', mu_0,...
%     'kappa_0', k_0, 'kappa_1', k_1, 'cov_shape', m, 'invg_shape', a_0, 'invg_scale', b_0, 'iter', 5);
%
% Ex: (unconstrained)
%     Bayesian_ZSL(Data split...,'Model','unconstrained', 'tuning', true, 'prior_mean', mu_0, 
%     'prior_cov', psi, 'kappa_0', k_0, 'kappa_1', k_1, 'cov_shape', m, 'pca', 500);

clear all;
%clc;

datapath = 'C:\Users\sbadirli\OneDrive - Indiana University\Desktop\ZSL data\';
dataset = 'INSECT 1BIN';

fname1=[datapath, dataset, '\res101.mat'];   
fname2=[datapath, dataset, '\splits_wo_att_1BIN.mat']; %'\att_splits.mat'];
load(fname1)
load(fname2)
load('Genus_ids_1BIN.mat')
load('Open_set_labels_1BIN.mat')
% load('NIPS_DNA_CNN_aligned_500_5e.mat');
load('DNA_CNN_embeddings_adam_5e_nofilter_noalignment_500_1BIN.mat');

dna_cnn = [dna_cnn, features'];

% load('DNA_CNN_embeddings_adam_5e_500.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  
% % Splitting the training data into training and test data for the use in
% %%%hyper-parameter tuning %%% DONT forget to change fname1 to fname2
[xtrain, ytrain, xtest_unseen, ytest_unseen, xtest_seen, ytest_seen] = data_split(dna_cnn', labels, trainval_loc, train_loc, ...
                                                                                 test_seen_loc, test_unseen_loc, val_seen_loc,...
                                                                                 val_unseen_loc, 'tuning');
% % % % % % 
ytest_unseen_new = G(ytest_unseen);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x_tr_g = V*features(:, train_loc);
% % %x_tr_g = V*dna_cnn(train_loc, :)';
% % 
% xtrain = [xtrain; x_tr_g'];
% ytrain = [ytrain; ytrain];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Tuning process and optimal parameters from CV
[k_0, k_1, m, a_0, b_0, mu_0, s, K] = hyperparameter_tuning(xtrain,ytrain,xtest_unseen,ytest_unseen_new,xtest_seen,ytest_seen,G,'Model', 'unconstrained', 'pca', 500);

% Splitting the data into train and test
% tr -- train, ts -- test, s -- seen, us -- unseen, x -- data matrix, 
% y -- labels 

% If you want to just reproduce the results from the paper use the following
% h-params settings from Supplementary materiall for a specific dataset
model_version = 'unconstrained';
pca_dim = 500;

%[K, k_0, k_1, m, s, a_0 , b_0] = load_tuned_params(dataset, model_version);
k0_range = [k_0]; %[0.1]; 
k1_range = [k_1]; %[10]; 
m_range = [m]; 
s_range = [s];
% a0_range = [1 10]; %[10]; 
% b0_range = [0.1 1]; %[1];
K=0;
best_h = 0;
best_s = 0;
best_us = 0;


[x_tr, y_tr, x_ts_us, y_ts_us, x_ts_s, y_ts_s] = data_split(dna_cnn', labels_ng, trainval_loc, train_loc, test_seen_loc,...
                                                            test_unseen_loc, val_seen_loc, val_unseen_loc, 'test');

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% x_tr_g = V*dna_cnn(trainval_loc, :)';

% %%%%%%%%%%%%%%%%%%%%%%%%%
C = cov(x_tr);
[vv, ~] = eig(C);
x_tr    = x_tr*vv(:,end-pca_dim+1:end);
x_ts_s  = x_ts_s*vv(:,end-pca_dim+1:end);
x_ts_us = x_ts_us*vv(:,end-pca_dim+1:end);
pca_dim = 0;

% train and test the data in GZSL setting
for k_0=k0_range
    for k_1=k1_range
        for m=m_range %a_0=a0_range%
            for s=s_range %b_0=b0_range%
                tic
                if strcmp(model_version, 'unconstrained')
                    [gzsl_seen_acc, gzsl_unseen_acc, H, s_cls_acc, us_cls_acc, pb_s, pb_us, class_id] = Bayesian_ZSL(x_tr, y_tr, x_ts_us, y_ts_us, x_ts_s, y_ts_s, G,'Model', model_version,'num_neighbor', K,...
                                                                   'kappa_0', k_0, 'kappa_1', k_1,'cov_shape', m,'prior_covscale',s, 'pca', pca_dim);
                     if  gzsl_seen_acc > best_s
                         best_s = gzsl_seen_acc;
                         best_seen_params = [k_0, k_1, m, s, K];
                     end

                     if  gzsl_unseen_acc > best_us
                         best_us = gzsl_unseen_acc;
                         best_unseen_params = [k_0, k_1, m, s, K];
                     end

                     if H>best_h
                         best_h = H;
                         best_params = [k_0, k_1, m, s, K];
                     end
                else
                    [gzsl_seen_acc, gzsl_unseen_acc, H] = Bayesian_ZSL(x_tr, y_tr, x_ts_us, y_ts_us, x_ts_s, y_ts_s, G,'Model', model_version,'num_neighbor', K,...
                                                                       'kappa_0', k_0, 'kappa_1', k_1, 'invg_shape', a_0, 'invg_scale', b_0);

                     if  gzsl_seen_acc > best_s
                         best_s = gzsl_seen_acc;
                         best_seen_params = [k_0, k_1, a_0, b_0];
                     end

                     if  gzsl_unseen_acc > best_us
                         best_us = gzsl_unseen_acc;
                         best_unseen_params = [k_0, k_1, a_0, b_0];
                     end

                     if H>best_h
                         best_h = H;
                         best_params = [k_0, k_1, a_0, b_0];
                     end
                end
                fprintf('Results from k0=%.2f, k1=%.2f, m=%d, s=%.1f\n', k_0, k_1, m, s);
                %fprintf('Results from k0=%.2f, k1=%.2f, a0=%d, b0=%d \n', k_0, k_1, a_0, b_0);
                fprintf('BZSL (%s version) IMAGE+DNA concat results on %s dataset: Seen acc: %.2f%% Unseen acc: %.2f%%, Harmonic mean: %.2f%% \n', ...
                        model_version, dataset, gzsl_seen_acc*100, gzsl_unseen_acc*100, H*100);
                toc
                
            end
        end
    end
end
