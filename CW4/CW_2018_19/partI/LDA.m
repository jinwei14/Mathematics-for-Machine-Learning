function U_reduc=LDA(X,class_label)

    X=X'; 
    %data centering using the smart way
    [~,n] = size(X);
    X_mean = X * (eye(n) - (1/n)*ones(n,1)*ones(n,1)');

    %compute M the mean matrix
    M=zeros(size(X_mean,2));
    n=1;
    classes=unique(class_label); 
    
    % get the number of labels/ class
    class_num=length(classes); 
    for i=1:class_num
        Nc_i=sum(class_label==classes(i)); 
        Ei=(1/Nc_i)*ones(Nc_i,Nc_i); 
        M(n:n+Nc_i-1,n:n+Nc_i-1)=Ei; 
        n=n+Nc_i;
    end
 
    %compute kw
    I=eye(size(X_mean,2));
    kw=(I-M)*(X_mean)'*X_mean*(I-M);
    kw=(1/2)*(kw+kw');
    %Sw=(I-M)*(X_bar)'*X_bar*(I-M);
    [Vw,Dw]=eigs(kw,size(X_mean,2)-class_num-1); 

    %computing U = X(I ? M)Vw?w?1
    U=X_mean*(I-M)*Vw*Dw^(-1);

    %compute Sb
    Sb=U'*X_mean*M*(X_mean)'*U; 

    [Q,D]=eigs(Sb,class_num-1); 
    [~,index]=sort(diag(D),'descend');

    Q=Q(:,index);
    U_reduc=U*Q;
end
