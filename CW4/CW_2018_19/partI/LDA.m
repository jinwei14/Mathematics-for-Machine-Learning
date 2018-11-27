% to be completed

% to be completed U_reduc = PCA(fea_Train,size(fea_Train,1)-1); 
function W = LDA(fea_Train,gnd_Train)
    X = fea_Train';
    label = gnd_Train';
    [~,p] = size(label);
    [~,n] = size(X);
%    diaArray = zeros(1,p); 
    M = zeros(n,n); 
    i = 1;
    while i<p
       number =  sum(gnd_Train(:) == label(i));
       E(1:number,1:number) = 1/number;
       M(i :i + number-1,i :i + number-1) = E;
       i = i + number;
    end
    

    %(I ? M)X^T X(I ? M) = V_w ?_w V_w^T 
    k_w = (eye(n) - M)* (X' * X) *(eye(n) - M);
    [V,A] = eig(0.5*(k_w*k_w'));
   
    % U = X(I ? M) Vw ?w?1
    U = X*(eye(n) - M)*V*A^(-1);
    X_head = U' * X * M;
    
    [Q,B] = eig(X_head*X_head');
%     [V,A] = eig(X_mean'* X_mean);
    [~,index] = sort(diag(B),'descend');
    Q = Q(:,index);
    
    %The total transform is W = UQ
    W =U*Q;
end
