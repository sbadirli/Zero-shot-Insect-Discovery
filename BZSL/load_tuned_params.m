% This function loads the hyperparameter set to reproduce results of the model specified.
% Please refer to the tuning  range and other tuning details to the Supp.
% Materials.

function [k_0, k_1, m, s] = load_tuned_params(model)
    
    model = upper(model);
    dim  = 500;
    OSBC_IMG  = [0.1, 10, 5*dim, 1];
    OSBC_DNA  = [0.1, 10, 25*dim, 0.5];
    OSBC_DIL  = [0.1, 10, 5*dim, 0.5];
    OSBC_DIT  = [0.1, 10, 25*dim, 0.5];
    eval(['data = ', model,';']);
    data = num2cell(data);
    [k_0, k_1, m, s] = deal(data{:});
end