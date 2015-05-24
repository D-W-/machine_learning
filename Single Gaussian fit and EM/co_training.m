%% co_training: use the co training method to process data
function [mark] = co_training(views,maxSelectNum)
    % ÿ�����ѡȡ2�����ݽ��б��
    % maxSelectNum = 2;
    viewNum = size(views,2);
    % m������ n������ֵ
    [m,n] = size(views{1});
    unlableNum = floor(m*0.2);
    lableNum = m - unlableNum;
    % ���,δ���������Ƿ���ĳ����ͼ���Ѿ���������
    mark = zeros(1,unlableNum);
    iter = 0;
    maxIteration = unlableNum;
    while iter < maxIteration
        for i = 1:viewNum
            % ���б�ǩ������
            train = views{i}(1:lableNum,:);
            test = views{i}(lableNum+1:m,:);
            test = test';
            % ��û�б�ǩ���������ҵ��Ѿ���ǵļ���train����
            for k = 1:unlableNum
                if mark(k) ~= 0
                    train = [train;views{i}(k+lableNum,:)];
                end
            end
            % Ϊ��ʹmatlab������Ƶ���ʽһ��,��Ҫת��һ��
            train = train';
            [mu,co,bundary] = gaussian(train);
            % ����δ���������
            store = (2*pi)^(-n/2)/sqrt(det(co));
            for k = 1:unlableNum
                p(k) = store*exp(-0.5*(test(:,k)-mu)'/co*(test(:,k)-mu));
            end
            % ѡȡ������Ŀ��δ�����������������
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
    % Ϊ������ϳ���һ����Ԫ��̬�ֲ�
    [n,m] = size(data);
    mu = sum(data,2)./m;
    temp = zeros(n,n);
    for i = 1:m
        temp = temp + (data(:,i)-mu)*(data(:,i)-mu)';
    end
    co = temp./m;

    % ʹ����ϳ����ĺ�������������ݷֲ������еĸ���
    store = (2*pi)^(-n/2)/sqrt(det(co));
    for i = 1:m
        p(i) = store*exp(-0.5*(data(:,i)-mu)'/co*(data(:,i)-mu));
    end

    bundary = min(p)*2;
