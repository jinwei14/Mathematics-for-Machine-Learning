function [ U_reduc ] = LDA_SSS( gnd_Train, fea_Train )

%%% LDA SSS
% Transpose training data. Find X which is an FxN matrix.
X = fea_Train';

% De-mean X.
X = bsxfun(@minus, X, mean(X));

% Find all unique labels, arranged in ascending order.
labels_unique = unique(gnd_Train);

% Count how many cases in gnd_Train have a particular label.
% Store these counts in labels_sum.
total_classes = length(labels_unique);
labels_sum = zeros(total_classes, 1);
for j = 1 : total_classes
    % Counts number of labels that are labels_unique(j) in gnd_Train.
    labels_sum(j) = sum(gnd_Train == labels_unique(j));
end

% Create block matrix whose diagonal entries are E_i's.
M = magic(0);
for j = 1 : total_classes
    count = labels_sum(j);
    E = (1 / count) * ones(count);
    M = blkdiag(M, E);
end

% Create matrix of data minus class mean.
X_W = X * (eye(size(M, 1)) - M);

% Compute eigenvalues and eigenvectors using svd.
[V_W, Lambda_W, ~] = svd((X_W)' * X_W);

% Select N - (C + 1) largest non-zero eigenvalues and eigenvectors.
N = size(X, 2);
rank_req = N - (total_classes + 1);

V_W = V_W(:, 1 : rank_req);
Lambda_W = Lambda_W(1 : rank_req, 1 : rank_req);

% Compute vector U with formula in lecture notes.
% Perform whitening on S_w = X(I - M) X^T.
U = X_W * V_W / (Lambda_W);

% Compute X_b.
X_b = U' * X * M;

% De-mean X_b
X_b = bsxfun(@minus, X_b, mean(X_b));

% Perform svd to obtain eigenvectors of X_b.
[Q, ~, ~] = svd(X_b * (X_b)');

% Obtain total transform.
U_reduc = U * Q;

end

