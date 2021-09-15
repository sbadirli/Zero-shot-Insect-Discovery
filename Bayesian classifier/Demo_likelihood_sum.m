
clear all;

%%% Loading the data %%%
datapath = '..\data\';
dataset = 'INSECTS';

fname1=[datapath, dataset, '\data.mat'];   
fname2=[datapath, dataset, '\splits.mat']; 
load(fname1)
load(fname2)
model = 'OSBC_DIL';

pca_dim = 500; 
[k_0, k_1, m, s] = load_tuned_params(model);
% Loading DNA data and training
[x_tr, y_tr, x_ts_us, y_ts_us, x_ts_s, y_ts_s] = data_split(embeddings_dna, embeddings_img, labels, trainval_loc, train_loc, test_seen_loc,... 
                                                            test_unseen_loc, val_seen_loc, val_unseen_loc, 'test', 'OSBC_DNA');

[seen_acc, unseen_acc, H, s_cls_acc, us_cls_acc, pb_s, pb_us, class_id] = Bayesian_cls(x_tr, y_tr, x_ts_us, y_ts_us, x_ts_s, y_ts_s, G,...
                                                                       'kappa_0', k_0, 'kappa_1', k_1,'cov_shape', m,'prior_covscale',s, 'pca', pca_dim);

fprintf('Bayesina classifer DNA INITIAL results on %s dataset: Seen acc: %.2f%% Unseen acc: %.2f%%, Harmonic mean: %.2f%% \n',dataset, seen_acc*100, unseen_acc*100, H*100);

% Loading IMAGE data and training
[x_tr, y_tr, x_ts_us, y_ts_us, x_ts_s, y_ts_s] = data_split(embeddings_dna, embeddings_img, labels, trainval_loc, train_loc, test_seen_loc,... 
                                                            test_unseen_loc, val_seen_loc, val_unseen_loc, 'test', 'OSBC_IMG');

[seen_acc, unseen_acc, H, s_cls_acc_i, us_cls_acc_i, pb_s_i, pb_us_i, class_id] = Bayesian_cls(x_tr, y_tr, x_ts_us, y_ts_us, x_ts_s, y_ts_s, G,...
                                                                       'kappa_0', k_0, 'kappa_1', k_1,'cov_shape', m,'prior_covscale',s, 'pca', pca_dim);
 

fprintf('Bayesian classifier IMAGE INITIAL results on %s dataset: Seen acc: %.2f%% Unseen acc: %.2f%%, Harmonic mean: %.2f%% \n',dataset, seen_acc*100, unseen_acc*100, H*100);


%%%%%%%%%%%%%%%%%%%%% Normalized summation of likelihoods %%%%%%%%%%%%%%%%%%%%%%
ntype = 'range';
P_us = normalize(pb_us, 2, ntype);
P_s = normalize(pb_s, 2, ntype);
P_us_i = normalize(pb_us_i, 2, ntype);
P_s_i = normalize(pb_s_i, 2, ntype);

PB_s = P_s + P_s_i;
PB_us = P_us + P_us_i;

[~, I_s] = sort(PB_s, 2, 'descend');
ypred_s = class_id(I_s(:, 1));

[~, I_us] = sort(PB_us, 2, 'descend');
ypred_us = class_id(I_us(:, 1));

[acc_per_usclass, unseen_acc] = evaluate(y_ts_us, ypred_us, G, 'unseen');
[acc_per_sclass, seen_acc] = evaluate(y_ts_s, ypred_s, G, 'seen');
H = 2*unseen_acc*seen_acc/(unseen_acc + seen_acc); 
fprintf('BZSL Likelihood-SUM results on %s dataset: Seen acc: %.2f%% Unseen acc: %.2f%%, Harmonic mean: %.2f%% \n',dataset, seen_acc*100, unseen_acc*100, H*100);
