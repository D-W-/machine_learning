%% co_training: use the co training method to process data
function [mark] = co_training(views,maxSelectNum)
    % 每次最多选取2组数据进行标记
    % maxSelectNum = 2;
    viewNum = size(views,2);
    % m个样本 n个特征值
    [m,n] = size(views{1});
    unlableNum = floor(m*0.2);
    lableNum = m - unlableNum;
    % 标记,未分类数据是否在某个视图下已经给出分类
    mark = zeros(1,unlableNum);
    iter = 0;
    maxIteration = unlableNum;
    while iter < maxIteration
        for i = 1:viewNum
            % 已有标签的数据
            train = views{i}(1:lableNum,:);
            test = views{i}(lableNum+1:m,:);
            test = test';
            % 在没有标签的数据中找到已经标记的加入train里面
            for k = 1:unlableNum
                if mark(k) ~= 0
                    train = [train;views{i}(k+lableNum,:)];
                end
            end
            % 为了使matlab代码和推导公式一致,需要转置一下
            train = train';
            [mu,co,bundary] = gaussian(train);
            % 计算未分类的数据
            store = (2*pi)^(-n/2)/sqrt(det(co));
            for k = 1:unlableNum
                p(k) = store*exp(-0.5*(test(:,k)-mu)'/co*(test(:,k)-mu));
            end
            % 选取合适数目的未标记样本加入标记样本
            [p,index] = sort(p);
            selected = 0;
            k = 1;
            while selected < maxSelectNum && k < unlableNum
                if mark(index(k)) == 0 && p(k) > bundary
                    mark(index(k)) = 1;
                    selected = selected +1;
                end
                k = k + 1;
            end
        end
        iter = iter + 1;
    end

function [mu,co,bundary] = gaussian(data)
    % 为数据拟合出来一个多元正态分布
    [n,m] = size(data);
    mu = sum(data,2)./m;
    temp = zeros(n,n);
    for i = 1:m
        temp = temp + (data(:,i)-mu)*(data(:,i)-mu)';
    end
    co = temp./m;

    % 使用拟合出来的函数计算测试数据分布在其中的概率
    store = (2*pi)^(-n/2)/sqrt(det(co));
    for i = 1:m
        p(i) = store*exp(-0.5*(data(:,i)-mu)'/co*(data(:,i)-mu));
    end

    bundary = min(p)*2;
