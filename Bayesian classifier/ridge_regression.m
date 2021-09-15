function V = ridge_regression(embeddings_dna, embeddings_img, data_subset, lambda)

di = size(embeddings_img, 2);
st = data_subset; 
X = embeddings_img(st, :)';
D = embeddings_dna(st, :)';

DXT = D*X';
V = DXT/(X*X' + lambda*eye(di));
end