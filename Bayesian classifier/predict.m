% This function calculates PPD (Student-t) for each class (seen and surroagte)
%
% Inputs:
%   Data -- X
%   Class specific PPD parameters -- Sig_s, mu_s, v_s, class_id
% Outputs:
%   labels -- ypred
%   likelihood probability matrix -- prob_mat

function [ypred, prob_mat] = predict(X,Sig_s,mu_s,v_s,class_id)

% Initialization
[ncl, d]   = size(mu_s);
piconst    = (d/2)*log(pi);
gl_pc      = gammaln(0.5:0.5:max(v_s)+d);
n          = size(X,1); 
prob_mat   = zeros(n,ncl);

% Calculating log student-t likelihood for numerical stability
for j=1:ncl
    v             = X - mu_s(j,:);         % Center the data
    chsig         = chol(Sig_s(:,:,j));    % Cholesky decomposition
    tpar          = gl_pc(v_s(j)+d)-(gl_pc(v_s(j))+(d/2)*log(v_s(j))+piconst)-sum(log(diag(chsig)));  % Stu-t lik part 1
    temp          = mrdivide(v,chsig);
    prob_mat(:,j) = tpar-0.5*(v_s(j)+d)*log(1+(1/v_s(j))*sum(temp.*temp,2));
end

[~, bb]    = max(prob_mat,[],2);
ypred      = class_id(bb);                 % To ensure labels are correctly assigned back to original ones
end