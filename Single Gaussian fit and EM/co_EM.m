%% co_EM: 多视图下的EM算法 参数views是不同视图下的样本数据 centerNum是聚类中心数目 maxIteration是最大循环次数 th收敛阈值
function [qi] = co_EM(views,centerNum,maxIteration,th)
    viewNum = size(views,2);
    % m个样本 n个特征值
    [m,n] = size(views{viewNum});
    % 先随机初始化每个样本分别属于每一类的概率
    qi = init(centerNum,m);

    % 当前循环次数
    iter = 0;
    % 初始化当前的首先程度
    t = 1;
    while iter < maxIteration && t > th
        for i = 1:viewNum
            % 获取训练数据
            train = views{i};
            [m,n] = size(train);
            train = train';
            % 多视图的EM算法必须先进行m步再进行e步,否则不同视图特征值维度不同无法计算
            % m步
            [lastFai,lastMu,lastCova] = m_step(qi,train,centerNum,m,n);
            % e步
            qi = e_step(train,lastFai,lastMu,lastCova,centerNum,m,n);
            % 保存一下
            fai = lastFai;
            mu = lastMu;
            cova = lastCova;
        end
        % 计算收敛程度
        t = max([norm(lastFai(:)-fai(:))/norm(lastFai(:));norm(lastMu(:)-mu(:))/norm(lastMu(:));norm(lastCova(:)-cova(:))/norm(lastCova(:))]);
        iter = iter + 1;
    end

%% init: 初始化qi,也就是每个样本属于每一类的概率 m是类别数目 n是样本数目
function [qi] = init(m,n)
    % 这里只是简单的随机初始化
    qi = rand(m,n);
    s = sum(qi);
    s = repmat(s,m,1);
    % 归一化
    qi = qi./s;

%% e_step: EM算法中的E步 x为数据 m个聚类中心 n个数据 k个特征值
function [qi] = e_step(x,fai,mu,cova,m,n,k)
    % 首先求一下每个样本点的概率,存在qi_p里面,方便后续计算
    % 外层循环:对于不同的高斯分布
    for i=1:m
        mu_i = mu(:,i);
        cov_i = cova(:,:,i);
        qi_p = zeros(m,n);
        % 内层循环,对于每个高斯分布中的每一个样本
        for j=1:n
            p_i = exp(-0.5*(x(:,j)-mu_i)'/cov_i*(x(:,j)-mu_i));
            qi_p(i,j) = p_i;
        end
        % 优化,对于同一个高斯分布,协方差矩阵-cov是相同的,所以拿到循环外面计算
        qi_p(i,:) = qi_p(i,:)/sqrt(det(cov_i));
    end
    % 优化,对于所有数据,特征值数-centerNum个聚类中心是相同的,拿到最外面计算
    qi_p = qi_p*(2*pi)^(-k/2);

    % 用全概率公式求qi
    % 除的是每一列的和,先把每个位置算出来,再复制一下行向量
    for i=1:m
        qi_t(i,:) = fai(i).*qi_p(i,:);
    end
    temp = sum(qi_t);
    temp = repmat(temp,m,1);
    qi = qi_t./temp;

%% m_step: EM算法中的M步 x为数据  m个聚类中心 n个数据 k个特征值
function [fai,mu,cova] = m_step(qi,x,m,n,k)
    mu = zeros(k,m);
    cova = zeros(k,k,m);
    % 为了方便计算,将多个更新公式中都会用到的一部分提取出来,记作temp
    temp = sum(qi,2)';
    % 更新fai
    fai = temp/n;
    % 更新mu
    for i=1:m
        sum_mu = 0;
        for j = 1:n
            sum_mu = sum_mu + qi(i,j)*x(:,j);
        end
        mu(:,i) = sum_mu/temp(i);
    end
    % 更新cov
    for i=1:m
        qi_sum_cov=zeros(k,k);
        for j=1:n
            qi_sum_cov = qi_sum_cov + qi(i,j)*(x(:,j)-mu(:,i))*(x(:,j)-mu(:,i))';
        end
        cova(:,:,i)=qi_sum_cov/temp(i);
    end


