% This function splits the training data into Train and Test data for the
% purpose of hyper-parameter tuning

function [xtrain, ytrain, xtest_unseen, ytest_unseen, xtest_seen, ytest_seen] = data_split(embeddings_dna, embeddings_img, labels, trainval_loc, train_loc, ...
                                                                                test_seen_loc, test_unseen_loc, val_seen_loc,...
                                                                                val_unseen_loc, split_type, model)
features = embeddings_dna;
if strcmp(model, 'OSBC_IMG')
    features = embeddings_img;
end
train_idx = trainval_loc;
test_seen_idx = test_seen_loc;
test_unseen_idx = test_unseen_loc;

if strcmp(split_type, 'tuning')
    train_idx = train_loc;
    test_seen_idx = val_seen_loc;
    test_unseen_idx = val_unseen_loc;
end

% Training data and labels
xtrain            = features(train_idx ,:);
ytrain            = labels(train_idx); 
% Test data and labels, Seen and Unseen
xtest_seen   = features(test_seen_idx,:); 
ytest_seen   = labels(test_seen_idx);

xtest_unseen = features(test_unseen_idx,:); 
ytest_unseen = labels(test_unseen_idx);

end




    





