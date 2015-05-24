function [out,k,COEFF] = dataProcess(filename)
% 数据处理,包括文件读取,PCA降维,降维后的数据存储
    % 文件处理 直接使用importdata命令可以提取出所有的数据,存在data字段中
    A = importdata(filename);
    A = A.data;
    % PCA对数据降维 COEFF是特征向量矩阵 latent代表贡献度
    [COEFF, SCORE, latent]=princomp(A);
    % 找出贡献度大于95%的特征值
    rate = cumsum(latent)./sum(latent);
    for k=1:size(rate)
        if rate(k)>0.95
            break;
        end
    end
    COEFF = COEFF(:,1:k);
    out = A*COEFF;
    filename1 = strcat(filename,'.mat');
    save(filename1,'out');
    filename2 = strcat(filename,'COEFF.mat');
    save(filename2,'COEFF');
end
