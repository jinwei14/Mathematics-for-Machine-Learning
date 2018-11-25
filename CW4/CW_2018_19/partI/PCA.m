% to be completed U_reduc = PCA(fea_Train,size(fea_Train,1)-1); 

function p = PCA(fea_Train,d)
    p = fea_Train*d;
end