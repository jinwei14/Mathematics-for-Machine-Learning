function [ U_reduc ] = PCA_SSS( fea_Train, Whitening )

% Input:
%    fea_Train: Training features, passed in from protocol.m.
%    Whitening: 0 or 1 flag where 0: no whitening; 1: with whitening.
% Output:
%    U_reduc: Returns projection matrix for PCA.

% bsxfun takes a binary function (@minus) and subtracts fea_Train from its
% mean. The result is the training data matrix with its mean removed.
X = bsxfun(@minus, fea_Train, mean(fea_Train));

% Transpose training data. Find X which is an FxN matrix.
X = X';
N = size(X, 2);

% Find dot product matrix, and perform eigen-analysis by using svd.
% Obtain V, eigenvectors and Lambdas, diagonal matrix of eigenvalues.
DotProduct = X' * X;
[V, Lambdas, ~] = svd(DotProduct);

% Select the N - 1 largest eigenvectors and eigenvalues.
V = V(:, 1 : N - 1);
Lambdas = Lambdas(1 : N - 1, 1 : N - 1);

% Take inverse square root of each element in the diagonal vector.
LambdasDiag = diag(Lambdas);
LambdasDiag = LambdasDiag .^ (-0.5);
Lambdas = diag(LambdasDiag);

% Compute eigenvectors of XX' using lemma from lectures.
U_reduc = X * V * Lambdas;

if Whitening
    U_reduc = U_reduc * Lambdas;
end


end
