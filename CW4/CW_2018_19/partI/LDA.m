% % to be completed
% 
% % to be completed U_reduc = PCA(fea_Train,size(fea_Train,1)-1); 
% function W = LDA(fea_Train,gnd_Train)
%     X = fea_Train';
%     label = gnd_Train';
%     [~,p] = size(label);
%     [~,n] = size(X);
%     classes=unique(gnd_Train); 
%     class_num=length(classes); 
% %    diaArray = zeros(1,p); 
%     M = zeros(n,n); 
%     i = 1;
%     while i<p
%        number =  sum(gnd_Train(:) == label(i));
%        E(1:number,1:number) = 1/number;
%        M(i :i + number-1,i :i + number-1) = E;
%        i = i + number;
%     end
%     
% 
%     %(I ? M)X^T X(I ? M) = V_w ?_w V_w^T 
%     k_w = (eye(n) - M)* (X' * X) *(eye(n) - M);
%     [V,A] = eigs(0.5*(k_w*k_w'), size(n)-class_num-1);
%    
%     % U = X(I ? M) Vw ?w?1
%     U = X*(eye(n) - M)*V*A^(-1);
%     X_head = U' * X * M;
%     
%     [Q,B] = eig(X_head*X_head');
% %     [V,A] = eig(X_mean'* X_mean);
%     [~,index] = sort(diag(B),'descend');
%     Q = Q(:,index);
%     
%     %The total transform is W = UQ
%     W =U*Q;
% end


function U_reduc=LDA(X,class_set)
X=X'; 
%data centering
meanValue=mean(X,2)*ones(1,size(X,2));
X_bar=X-meanValue; 
 
%compute M compute Ei
M=zeros(size(X_bar,2));
n=1;
classes=unique(class_set); 
class_num=length(classes); 
for i=1:class_num
    Nc_i=sum(class_set==classes(i)); 
    Ei=(1/Nc_i)*ones(Nc_i,Nc_i); 
    M(n:n+Nc_i-1,n:n+Nc_i-1)=Ei; 
    n=n+Nc_i;
end
 
%compute kw
I=eye(size(X_bar,2));
kw=(I-M)*(X_bar)'*X_bar*(I-M);
kw=(1/2)*(kw+kw');
%Sw=(I-M)*(X_bar)'*X_bar*(I-M);
[Vw,Dw]=eigs(kw,size(X_bar,2)-class_num-1); 
 
%compute U
U=X_bar*(I-M)*Vw*Dw^(-1);
 
%compute Sb
Sb=U'*X_bar*M*(X_bar)'*U; 

[Q,D]=eigs(Sb,class_num-1); 
[D,index]=sort(diag(D),'descend');
D=D(index);
Q=Q(:,index);
U_reduc=U*Q;
end
