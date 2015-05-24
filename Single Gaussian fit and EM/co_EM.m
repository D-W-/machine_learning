%% co_EM: ����ͼ�µ�EM�㷨 ����views�ǲ�ͬ��ͼ�µ��������� centerNum�Ǿ���������Ŀ maxIteration�����ѭ������ th������ֵ
function [qi] = co_EM(views,centerNum,maxIteration,th)
    viewNum = size(views,2);
    % m������ n������ֵ
    [m,n] = size(views{viewNum});
    % �������ʼ��ÿ�������ֱ�����ÿһ��ĸ���
    qi = init(centerNum,m);

    % ��ǰѭ������
    iter = 0;
    % ��ʼ����ǰ�����ȳ̶�
    t = 1;
    while iter < maxIteration && t > th
        for i = 1:viewNum
            % ��ȡѵ������
            train = views{i};
            [m,n] = size(train);
            train = train';
            % ����ͼ��EM�㷨�����Ƚ���m���ٽ���e��,����ͬ��ͼ����ֵά�Ȳ�ͬ�޷�����
            % m��
            [lastFai,lastMu,lastCova] = m_step(qi,train,centerNum,m,n);
            % e��
            qi = e_step(train,lastFai,lastMu,lastCova,centerNum,m,n);
            % ����һ��
            fai = lastFai;
            mu = lastMu;
            cova = lastCova;
        end
        % ���������̶�
        t = max([norm(lastFai(:)-fai(:))/norm(lastFai(:));norm(lastMu(:)-mu(:))/norm(lastMu(:));norm(lastCova(:)-cova(:))/norm(lastCova(:))]);
        iter = iter + 1;
    end

%% init: ��ʼ��qi,Ҳ����ÿ����������ÿһ��ĸ��� m�������Ŀ n��������Ŀ
function [qi] = init(m,n)
    % ����ֻ�Ǽ򵥵������ʼ��
    qi = rand(m,n);
    s = sum(qi);
    s = repmat(s,m,1);
    % ��һ��
    qi = qi./s;

%% e_step: EM�㷨�е�E�� xΪ���� m���������� n������ k������ֵ
function [qi] = e_step(x,fai,mu,cova,m,n,k)
    % ������һ��ÿ��������ĸ���,����qi_p����,�����������
    % ���ѭ��:���ڲ�ͬ�ĸ�˹�ֲ�
    for i=1:m
        mu_i = mu(:,i);
        cov_i = cova(:,:,i);
        qi_p = zeros(m,n);
        % �ڲ�ѭ��,����ÿ����˹�ֲ��е�ÿһ������
        for j=1:n
            p_i = exp(-0.5*(x(:,j)-mu_i)'/cov_i*(x(:,j)-mu_i));
            qi_p(i,j) = p_i;
        end
        % �Ż�,����ͬһ����˹�ֲ�,Э�������-cov����ͬ��,�����õ�ѭ���������
        qi_p(i,:) = qi_p(i,:)/sqrt(det(cov_i));
    end
    % �Ż�,������������,����ֵ��-centerNum��������������ͬ��,�õ����������
    qi_p = qi_p*(2*pi)^(-k/2);

    % ��ȫ���ʹ�ʽ��qi
    % ������ÿһ�еĺ�,�Ȱ�ÿ��λ�������,�ٸ���һ��������
    for i=1:m
        qi_t(i,:) = fai(i).*qi_p(i,:);
    end
    temp = sum(qi_t);
    temp = repmat(temp,m,1);
    qi = qi_t./temp;

%% m_step: EM�㷨�е�M�� xΪ����  m���������� n������ k������ֵ
function [fai,mu,cova] = m_step(qi,x,m,n,k)
    mu = zeros(k,m);
    cova = zeros(k,k,m);
    % Ϊ�˷������,��������¹�ʽ�ж����õ���һ������ȡ����,����temp
    temp = sum(qi,2)';
    % ����fai
    fai = temp/n;
    % ����mu
    for i=1:m
        sum_mu = 0;
        for j = 1:n
            sum_mu = sum_mu + qi(i,j)*x(:,j);
        end
        mu(:,i) = sum_mu/temp(i);
    end
    % ����cov
    for i=1:m
        qi_sum_cov=zeros(k,k);
        for j=1:n
            qi_sum_cov = qi_sum_cov + qi(i,j)*(x(:,j)-mu(:,i))*(x(:,j)-mu(:,i))';
        end
        cova(:,:,i)=qi_sum_cov/temp(i);
    end


