% 对题目数据进行聚类的脚本
% 读入下载的文件数据,并输出测试结果

files = dir('data\\*.txt');
% 视图数目就是data下面的文件数目
viewNum = length(files);
% 将所有处理过的数据存在元组里面
views = cell(1,viewNum);
for i = 1:viewNum
    % 数据降维
    [out,k,COEFF] = dataProcess(strcat('data\\',files(i).name));
    views{i} = out;
end
% 存成文件备份一下
save('views.mat','views');


% 进行聚类,结果存在result里面
% result是1*24的矩阵,是对24个测试样本的分类输出,若认为患该种癌症,对应值为1
result = co_training(views,2);

disp('最终对测试样本的分类:');
disp(result);

disp('分类准确率:');
disp(sum(result)/length(result));

