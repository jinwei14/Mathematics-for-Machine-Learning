% to be completed

% to be completed U_reduc = PCA(fea_Train,size(fea_Train,1)-1); 
function p = wPCA(fea_Train,d)

    X = fea_Train';
    [~,n] = size(X);
    X_mean = X * (eye(n) - (1/n)*ones(n,1)*ones(n,1)');
    
    [V,A] = eig(X_mean'* X_mean);
    [~,index] = sort(diag(A),'descend');
    A = A(index,index);
    V = V(:,index);
    U = X * V * A^(-1);
%     size(U)
    U_d = U(:,1:d);

    p=U_d;
end
