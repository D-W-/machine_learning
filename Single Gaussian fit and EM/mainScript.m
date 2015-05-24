% ����Ŀ���ݽ��о���Ľű�
% �������ص��ļ�����,��������Խ��

files = dir('data\\*.txt');
% ��ͼ��Ŀ����data������ļ���Ŀ
viewNum = length(files);
% �����д���������ݴ���Ԫ������
views = cell(1,viewNum);
for i = 1:viewNum
    % ���ݽ�ά
    [out,k,COEFF] = dataProcess(strcat('data\\',files(i).name));
    views{i} = out;
end
% ����ļ�����һ��
save('views.mat','views');


% ���о���,�������result����
% result��1*24�ľ���,�Ƕ�24�����������ķ������,����Ϊ�����ְ�֢,��ӦֵΪ1
result = co_training(views,2);

disp('���նԲ��������ķ���:');
disp(result);

disp('����׼ȷ��:');
disp(sum(result)/length(result));

