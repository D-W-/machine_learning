function [out,k,COEFF] = dataProcess(filename)
% ���ݴ���,�����ļ���ȡ,PCA��ά,��ά������ݴ洢
    % �ļ����� ֱ��ʹ��importdata���������ȡ�����е�����,����data�ֶ���
    A = importdata(filename);
    A = A.data;
    % PCA�����ݽ�ά COEFF�������������� latent�����׶�
    [COEFF, SCORE, latent]=princomp(A);
    % �ҳ����׶ȴ���95%������ֵ
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
