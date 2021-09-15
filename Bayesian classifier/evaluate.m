function [acc_per_class, acc] = evaluate(Y_true, Y_pred, G, class_type)
if strcmp(class_type, 'unseen')
    Y_true = G(Y_true);
end
uy = unique(Y_true);
nc = length(uy);
acc_per_class   = zeros(nc, 1);

for i=1:nc
    idx = Y_true==uy(i);
    acc_per_class(i) = sum(Y_true(idx) == Y_pred(idx)) / sum(idx);
end

acc = mean(acc_per_class);
end