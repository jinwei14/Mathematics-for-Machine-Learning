% to be completed U_reduc = PCA(fea_Train,size(fea_Train,1)-1); 
function p = PCA(fea_Train,d)
    
    X = fea_Train';
    [~,n] = size(X);
    % Data Centering using Matrix Multiplication
    X_mean = X * (eye(n) - (1/n)*ones(n,1)*ones(n,1)');
    
    %Eigenanalysis
    [V,A] = eig(X_mean'* X_mean);
    
    %sort the eigrn value matrix convariance matrix
    [~,index] = sort(diag(A),'descend');
    
    %find the corresponding matrix
    A = A(index,index);
    V = V(:,index);
    
    %Compute eigenvectors:
    U = X * V * A^(-0.5);
    
    %Keep specific number of first components:
    U_d = U(:,1:d);

    p=U_d;
end
