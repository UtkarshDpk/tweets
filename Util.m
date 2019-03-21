%vmat = floor(10*rand([100,50]));
load dat_read_1year_dataset_5
vmat = vmat(2:end,:);
[totalHashes,totalTime]=size(vmat);

%% Mean Activity with Time(Global) of Hashtag
% dim 1 = column
% m=mean(vmat,1);
% figure(1)
% plot(m)
% grid on
% xlabel('Time(Days)'),ylabel('Mean Activity')
% set(findall(gca, 'Type', 'Line'),'LineWidth',2);
%% Mean Activity with Age of Hashtag
% age = zeros(totalHashes,1);
% 
% for i=1:totalHashes
%     age(i) = find(vmat(i,:),1);
% end
% 
% [totNewHashes,~]=size(find(age<=floor(totalTime/2)));
% newData=zeros(totNewHashes,totalTime-(min(age)-1));
% j=1;
% for i=1:totalHashes
%     if(age(i)<=floor(totalTime/2))
% %         if(birth(i)>1)
%             newData(j,:) = [vmat(i,age(i):end) zeros(1,age(i)-1)];
% %         else
% %             newData(j,:) = vmat(i,birth(i):end);
%         j=j+1;
% %         end
%     end
% end
% m=mean(newData,1);
% figure(2)
% plot(m)
% grid on
% xlabel('Time(Age of Hashtag)'),ylabel('Mean Activity')
% set(findall(gca, 'Type', 'Line'),'LineWidth',2);
%% Popularity at age 100
% pop100 = sum(newData(:,1:100),2);
% n=floor( log10(pop100));
% % order of popularity at age 100 is n
% pop1 = pop100(n==2);
% pop2 = pop100(n==3);
% pop3 = pop100(n==4);
% hash1 = zeros(10,100);
% hash2 = hash1;
% hash3 = hash1;
% pp1 = hash1;
% pp2 = pp1;
% pp3 = pp1;
% for i=1:10
% y1 = pop1(randi(numel(pop1)));
% y2 = pop2(randi(numel(pop2)));
% y3 = pop3(randi(numel(pop3)));
% hash1(i,:) = newData(find(pop100==y1,1),1:100);
% hash2(i,:) = newData(find(pop100==y2,1),1:100);
% hash3(i,:) = newData(find(pop100==y3,1),1:100);
% pp1(i,:) = cumsum(hash1(i,:));
% pp2(i,:) = cumsum(hash2(i,:));
% pp3(i,:) = cumsum(hash3(i,:));
% figure(3)
% plot(1:100,pp1(i,:))
% grid on
% hold on
% xlabel('Age of Hashtag)'),ylabel('Popularity')
% set(findall(gca, 'Type', 'Line'),'LineWidth',2);
% figure(4)
% plot(1:100,pp2(i,:))
% grid on
% hold on
% xlabel('Age of Hashtag)'),ylabel('Popularity')
% set(findall(gca, 'Type', 'Line'),'LineWidth',2);
% figure(5)
% plot(1:100,pp3(i,:))
% grid on
% hold on
% xlabel('Age of Hashtag)'),ylabel('Popularity')
% set(findall(gca, 'Type', 'Line'),'LineWidth',2);
% end
% hold off
%% Moving Average of Tweets
% hashtagNum = 1;
% m3 = meanActivityForHashtag(hashtagNum,totalTime,vmat);
% figure(4)
% plot(m3);
% grid on
% xlabel('Day'),ylabel('Moving Average of Tweets')
% title(sprintf('Hashtag Number:%d',hashtagNum));

%% Poplularity of Hashtag
% dim 2 = row
popularity = sum(vmat,2);
% [N] = histcounts(popularity);
% N1 = sort(N,'descend');
% figure(10)
% [slope, intercept] = logfit(popularity,N,'loglog'); 
% yApprox = (10^intercept)*x.^(slope);
% plot(1:numel(N),N./N(1),'LineWidth',2)
% grid on
% % set(gca,'YScale','log');
% set(gca,'YScale','log','XScale','log');
% xlabel('Popularity')
% ylabel('Probability of Popularity')
% paramEsts = gpfit(popularity);
% k = paramEsts(1);% shape param
% sigma = paramEsts(2); % scale param
% cd = gpcdf(popularity,k,sigma,0,'upper');
% [m,v] = gpstat(k,sigma,0);
% figure(7)
% plot(popularity,cd)
% grid on
% xlabel('Popularity')
% ylabel('CCDF')
% set(gca,'YScale','log','XScale','log');
% figure(8)
% histogram(popularity);
% grid on
% xlabel('Popularity')
% set(gca,'YScale','log','XScale','log')
% pd = gppdf(popularity,k,sigma,0);
% figure(9)
% qqplot(popularity,pd);
% grid on

% pd = paretotails(popularity,0,0.5)
% [N,Edges] = histcounts(logPop);
% cumsumN = cumsum(N);
% fracN = cumsumN./cumsumN(end);
% line(linspace(0,6,numel(fracN)),1-fracN);
% h = findobj(gca,'Type','patch');
% h.FaceColor = [.8 .8 1];
% xgrid = linspace(2,3.9020e+05,1000);
% line(xgrid,.5*length(popularity)*gppdf(xgrid,paramEsts(1),paramEsts(2),0));

% grid on

%% Parameter Estimation
% rand creates uniformly distributed values between 0,1
% submat = vmat(popularity>1000,:);    
% [hashes,days] = size(submat);
% dup = zeros(hashes,days);
% ParamsExp = zeros(100,3);
% ParamsPow = zeros(100,4);
% K = 24*60*60; % No. of seconds in a day
for i=1
%     T = 0;
%     for j=1:days
%         retweets = submat(i,j);
%         if(retweets > 0)
%             u = rand(retweets,1);
%             nonZero = u(u ~= 0);
%             anyDuplicates = length(unique(nonZero)) ~= length(nonZero);
%             if(anyDuplicates ~= 0)
%                 dup(i,j) = warning('Duplicate time found at row and column %s,%s',i,j);
%             end
%             U = K*u + K*(j-1);
%             T = cat(1,T,U);
%         end
%     end
%     T = sort(T);
%     f = @(x)nloglf(T,x); 
%     % exponential kernel params=alpha,beta,lambda
%     
%     alpha = 1;
%     beta = 1.1;
%     lambda = 1;
%     
%     x0 = [alpha beta lambda];
%     lowerBound = [0 0 0];
%     [xmin,fmin,LL] = runfmincon(f,x0,lowerBound,[],[]);
%     ParamsExp(i,:) = xmin;
%     [condIntExp] = calculateLambdaStarExp(xmin,T);
%     plotLogLikelihoodExp(T,xmin,fmin);

    [xminPow,LLPow,fmin] = plotCondIntensityForPowerLaw(T(1:500));
    ParamsPow(i,:) = xminPow;
    calculateLambdaStarPow(xminPow,T,[]);
    plotLogLikelihoodPow(T,xminPow,fmin)
    
%     [Ts,condIntStar,condIntTemp] = simulateHawkesExp(10^3,ParamsExp);
%     plotCondInt(condIntStar,ParamsExp);
%     [condIntExpSim] = calculateLambdaStarExp(xmin,T);
end
% h1 = histogram(ParamsExp(:,1));
% hold on
% h2 = histogram(ParamsExp(:,2));
% h3 = histogram(ParamsExp(:,3));

%% Simulation of events
% function [Ts,condIntMax,condInt] = simulateHawkesExp(Tmax,ParamsExp)
% % Ogata's modified thinning
% alpha = ParamsExp(1,1)*10^5;
% beta = ParamsExp(1,2)*10^5;
% lambda = ParamsExp(1,3)*10^5;
% % condIntStar = ParamsExp(1,3);
% % u = rand;
% % tau = -log(u)/condIntStar;
% % while(s>T(end))
% %     u = rand;
% %     tau = -log(u)/condIntStar;
% % end
% % if(tau<=T(end))
% %     Ts(1) = tau;
% % end
% numOfSmallIntervals = 100;
% sum1 = 0;
% sum2 = 0;
% s = 0; n = 2;
% Ts = zeros(10000,1);
% A = zeros(size(Ts));
% condIntMax(n-1) = lambda;
% % Initiate cond int
% condInt(n-1) = condIntMax(n-1);
% % plot([Ts(n-1) Ts(n)],[condIntMax(n-1) condIntMax(n-1)],'k--');
% % plot([Ts(n-1) Ts(n)],[condInt(n-1) condInt(n-1)],'k-');
% while (s < Tmax)
%     % New Upper Bound
%     A(n) = exp(-beta*(s-Ts(n-1))).*(1+A(n-1));
%     condIntMax(n) = lambda + alpha*A(n);
%     % Initiate cond int
%     condInt(n) = condIntMax(n);
%     % Generate Next Point
%     u = rand;
%     tau = -log(u)/condIntMax(end);
%     s = s + tau;
%     D = rand;
%     plot(s,D,'b^');
%     grid on
%     hold on
%     condInt(n) = lambda + alpha*(exp(-beta.*(s-Ts(n-1)))).*(1+A(n-1));
%     if(D*condIntMax(n) <= condInt(n))
%         Ts(n) = s;
%         t(:,n-1) = linspace(Ts(n-1),Ts(n),numOfSmallIntervals)';
%         y = D*condIntMax(n);
%         plot(s,y,'bx');
%         condInt(:,n) = lambda*ones(numOfSmallIntervals,1)...
%             + alpha*(exp(-beta.*(t(:,n-1)-Ts(n-1)))).*(1+exp(-beta.*(t(:,n-2)-Ts(n-2))));
%         plot([Ts(n-1) Ts(n)],[condIntMax(n-1) condIntMax(n-1)],'k--');
%         plot(t(:,n-1),condInt(:,n),'k-');
%         n = n + 1;
%     else
%         plot(s,D*condIntMax(n-1),'bo');
%     end
% end
% plot(Ts,1,'-.');
% hold off
% if(Ts(end)<=Tmax(end))
%     return;
% else
%     Ts = Ts(1:end-1);
% end
% end
%% Functions
function [conditionalIntensity] = calculateLambdaStarExp(xmin,T)
alpha = xmin(1);
beta = xmin(2);
lambda = xmin(3);
numOfEventTimes = 10;
numOfSmallIntervals = 100;
conditionalIntensity = zeros(numOfSmallIntervals,numOfEventTimes);
conditionalIntensity(:,1) = lambda*ones(numOfSmallIntervals,1);
t = zeros(numOfSmallIntervals,numOfEventTimes);
t(:,1) = linspace(T(1),T(2),numOfSmallIntervals)';
figure(9)
plot(t,conditionalIntensity(:,1),'r-','LineWidth',2);
hold on
sum = 0;
for i = 2:numOfEventTimes
    t(:,i) = linspace(T(i),T(i+1),numOfSmallIntervals)';
    for j = 1:i-1
        sum = sum + alpha*exp(-beta * (t(:,i) - T(j)));
    end
    conditionalIntensity(:,i) = lambda + sum;
    plot([t(end,i-1) t(1,i)],[conditionalIntensity(end,i-1)...
        conditionalIntensity(1,i)],'ro--','LineWidth',2);
    p1 = plot(t(:,i),conditionalIntensity(:,i),'r-','LineWidth',2);
end
hold off
grid on
xlabel('Time t(sec)');ylabel('lambda*(t)');
% expectedIntensity = lambda/(1-alpha/beta);
% branchingRatio = alpha/beta;
end

function [xmin,LL,fmin] = plotCondIntensityForPowerLaw(T)
% Plot and Results
g = @(x)nloglfPow(T,x);
% PwerLaw kernel params c0,s0,p,lambda (take values from Kobayashi)
c0 = 6.49*10^-4;
s0 = 300;
p = 0.242;
lambda = 0.5;
x0 = [c0 s0 p lambda];
lowerBound = [0 0 -inf 0];
[xmin,fmin,LL] = runfmincon(g,x0,lowerBound,[],[]);% [0 0 1 0; 0 0 -1 0],[2;-1]

end

function [conditionalIntensity] = calculateLambdaStarPow(xmin,T,p1)
c0 = xmin(1);
s0 = xmin(2);
p = xmin(3);
lambda = xmin(4);
numOfEventTimes = 10;
numOfSmallIntervals = 100;
conditionalIntensity = zeros(numOfSmallIntervals,numOfEventTimes);
conditionalIntensity(:,1) = p*ones(numOfSmallIntervals,1);
t = zeros(numOfSmallIntervals,numOfEventTimes);
t(:,1) = linspace(T(1),T(2),numOfSmallIntervals)';
figure(10)
plot(t,conditionalIntensity(:,1),'r-','LineWidth',2);
hold on
sum = 0;
for i = 2:numOfEventTimes
    t(:,i) = linspace(T(i),T(i+1),numOfSmallIntervals)';
    for j = 1:i-1
        sum = sum + c0*((t(:,i) - T(j))./s0).^(-(1+p));
    end
    conditionalIntensity(:,i) = lambda + sum;
    plot([t(end,i-1) t(1,i)],[conditionalIntensity(end,i-1)...
        conditionalIntensity(1,i)],'ro--','LineWidth',2);
    p2 = plot(t(:,i),conditionalIntensity(:,i),'r-','LineWidth',2);
end
hold off
grid on
xlabel('Time t(sec)');ylabel('lambda*(t)');
% title('Conditional Intensity Function for different Kernels');
% legend([p1 p2],{'Exponential','Power-Law'},'Location','northwest')
end

function [] = plotLogLikelihoodExp(T,xmin,fmin)
    alp = linspace(xmin(1)/100,0.0005);
    bta = linspace(xmin(2)/100,0.0001);
    lbda = linspace(xmin(3)/100,0.0001);
    
    logLike1 = zeros(numel(alp),1);
    logLike2 = zeros(numel(bta),1);
    logLike3 = zeros(numel(lbda),1);
    for j = 1:numel(alp)
        x = [alp(j) xmin(2) xmin(3)];
        logLike1(j) = -nloglf(T,x);
    end
    figure(1)
    plot(alp,logLike1);
    hold on 
    grid on
    plot(xmin(1),-fmin,'*','LineWidth',2); hold off
    xlabel('alpha');ylabel('Log-Likelihood');
    title('Log-Likelihood vs alpha using Exponential kernel');
    
    for j = 1:numel(bta)
        x = [xmin(1) bta(j) xmin(3)];
        logLike2(j) = -nloglf(T,x);
    end
    figure(2)
    plot(bta,logLike2);
    hold on 
    grid on
    plot(xmin(2),-fmin,'*','LineWidth',2); hold off
    xlabel('beta');ylabel('Log-Likelihood');
    title('Log-Likelihood vs beta using Exponential kernel');
    
    for j = 1:numel(lbda)
        x = [xmin(1) xmin(2) lbda(j)];
        logLike3(j) = -nloglf(T,x);
    end
    figure(3)
    plot(lbda,logLike3);
    hold on 
    grid on
    plot(xmin(3),-fmin,'*','LineWidth',2); hold off
    xlabel('lambda');ylabel('Log-Likelihood');
    title('Log-Likelihood vs lambda using Exponential kernel');
end

function [] = plotLogLikelihoodPow(T,xmin,fmin)
par1 = linspace(xmin(1)/10,xmin(1)*10);
par2 = linspace(xmin(2)/10,xmin(2)*10);
par3 = linspace(xmin(3)/10,xmin(3)*10);
par4 = linspace(xmin(4)/10,xmin(4)*10);

logLike1 = zeros(numel(par1),1);
logLike2 = zeros(numel(par2),1);
logLike3 = zeros(numel(par3),1);
logLike4 = zeros(numel(par4),1);

for j = 1:numel(par1)
    x = [par1(j) xmin(2) xmin(3) xmin(4)];
    logLike1(j) = -nloglfPow(T,x);
end
figure(5)
plot(par1,logLike1);
hold on
grid on
plot(xmin(1),-fmin,'*','LineWidth',2); hold off
xlabel('c0');ylabel('Log-Likelihood');
title('Log-Likelihood vs c0 using Power-Law kernel');
for j = 1:numel(par2)
    x = [xmin(1) par2(j) xmin(3) xmin(4)];
    logLike2(j) = -nloglfPow(T,x);
end
figure(6)
plot(par2,logLike2);
    hold on
grid on
    plot(xmin(2),-fmin,'*','LineWidth',2); hold off
xlabel('s0');ylabel('Log-Likelihood');
title('Log-Likelihood vs s0 using Power-Law kernel');
for j = 1:numel(par3)
    x = [xmin(1) xmin(2) par3(j) xmin(4)];
    logLike3(j) = -nloglfPow(T,x);
end
figure(7)
plot(par3,logLike3);
    hold on
grid on
    plot(xmin(3),-fmin,'*','LineWidth',2); hold off
xlabel('p');ylabel('Log-Likelihood');
title('Log-Likelihood vs p using Power-Law kernel');
for j = 1:numel(par4)
    x = [xmin(1) xmin(2) xmin(3) par4(j)];
    logLike4(j) = -nloglfPow(T,x);
end
figure(8)
plot(par4,logLike4);
    hold on
grid on
    plot(xmin(4),-fmin,'*','LineWidth',2); hold off
xlabel('lambda');ylabel('Log-Likelihood');
title('Log-Likelihood vs lambda using Power-Law kernel');
end
% function [grad] = gradientOfExpKernelLogL(T)
% syms A(bta,a,b)
% syms sum1(lmbda,alfa,bta,a,b)
% syms sum2(alfa,bta,a,b)
% syms f(lmbda,a)
% syms g(alfa,bta,a,b)
%
% syms sum3(labda,bta) sum4(labda,alfa,bta)
% sum3(labda,bta,a,b) = 1./labda;
% sum4(labda,alfa,bta,a,b) = 0;
% syms sum5(bta,a,b) sum6(alfa,bta,lamda,a,b)
% sum5(bta,a,b) = exp(-bta.*(subs(b,T(end))-subs(a,T(1))))-1;
% sum6(alfa,bta,lamda,a,b) = 0;
% syms sum7(bta,a,b) sum8(bta,a,b)
% sum7(bta,a,b) = (1./bta^2).*exp(-bta.*(subs(b,T(end))-subs(a,T(1))));
% sum8(bta,a,b) = (1./bta).*(subs(b,T(end))-subs(a,T(1))).*exp(-bta.*(subs(b,T(end))-subs(a,T(1))));
%
% syms B(bta,a,b)
% B(bta,a,b) = 0;
% A(bta,a,b) = 0;
% sum1(lmbda,alfa,bta,a,b) = log(lmbda + alfa.*A(bta,a,b));
% sum2(alfa,bta,a,b) = exp(-bta.*(subs(a,T(end))-subs(b,T(1))))-1;
% f(lmbda,a) = lmbda.*subs(a,T(end));
% g(alfa,bta,a,b) = (alfa./bta).*sum1(lmbda,alfa,bta,a,b);
% for i=2:floor(length(T)/100000)
%     A(bta,a,b) = exp(-bta.*(subs(a,T(i))-subs(b,T(i-1)))).*(1+(A(bta,a,b)));
%     sum1(lmbda,alfa,bta,a,b) = sum1(lmbda,alfa,bta,a,b) + log(lmbda + alfa.*A(bta,a,b));
%     sum2(alfa,bta,a,b) = sum2(alfa,bta,a,b) + (exp(-bta.*(subs(a,T(end))-subs(b,T(i))))-1);
%
%     sum3(labda,bta,a,b) = sum3(labda,bta,a,b) + 1./(labda + alfa.*A(bta,a,b));
%     sum4(labda,alfa,bta,a,b) = sum4(labda,alfa,bta,a,b) + A(bta,a,b)./(labda + alfa.*A(bta,a,b));
%     sum5(bta,a,b) = sum5(bta,a,b) + exp(-bta.*(subs(b,T(end))-subs(a,T(i))))-1;
%     for j=1:i-1
%            B(bta,a,b) = B(bta,a,b) + (subs(b,T(end))-subs(a,T(i))).*(exp(-bta.*(subs(b,T(end))-subs(a,T(i)))));
%     end
%     sum6(alfa,bta,lamda,a,b) = sum6(alfa,bta,lamda,a,b) + alfa.*B(bta,a,b)./(labda + alfa.*(A(bta,a,b)));
%     sum7(bta,a,b) = sum7(bta,a,b) + (1./bta^2).*exp(-bta.*(subs(b,T(end))-subs(a,T(i))));
%     sum8(bta,a,b) = sum8(bta,a,b) + (1./bta).*(subs(b,T(end))-subs(a,T(i))).*exp(-bta.*(subs(b,T(end))-subs(a,T(i))));
% end
% nllf = -sum1(lmbda,alfa,bta,a,b) + f(lmbda,a) - g(alfa,bta,a,b);
%
% [solx,soly,solz] = solve(- sum4(labda,alfa,bta,a,b) - sum5(bta,a,b) == 0,...
%     alfa.*sum7(bta,a,b) + alfa.*sum8(bta,a,b) + sum6(alfa,bta,lamda,a,b) == 0, sum3(labda,bta,a,b) == T(end));
% end

function [meanvmatct] = meanActivity(vmat,r,c)
meanvmatct = zeros(r,c);
m2 = zeros(c,1);
for i=1:r
    for j=1:c
        m2(j) = sum(vmat(i,1:j))/j;
    end
    meanvmatct(i,:) = m2;
end
end

function [m3] = meanActivityForHashtag(hashtagNum,c,vmat)
m3 = zeros(c,1);
for j = 1:c
    m3(j) = sum(vmat(hashtagNum,1:j))/j;
end
end

% rng default % For reproducibility
    % gs = GlobalSearch;
    % ms = MultiStart;
%     opts = optimoptions(@fmincon,'Algorithm','interior-point','SubproblemAlgorithm','cg'...
%         ,'Display', 'iter-detailed','Diagnostics', 'on','SpecifyObjectiveGradient',true...
%         ,'HessianApproximation','bfgs','MaxIter',7);% 'OutputFcn',@plotLLAtIter
%     problem = createOptimProblem('fmincon','x0',x0,'objective',f,'options',opts);
%     [xmin,fmin,flag,outpt,allmins] = run(gs,problem);
    % flag 2 means At least one local minimum found. Some runs of the local solver converged.
    %     figure(2)
    %     plot(LL.x(:,1),LL.fval,'r*');
    %     xlabel('alpha');
    %     title('Log Likelihood');
    %     figure(3)
    %     plot(LL.x(:,2),LL.fval,'bo');
    %     xlabel('beta');
    %     title('Log Likelihood');
    %     figure(4)
    %     plot(LL.x(:,3),LL.fval,'cv');
    %     xlabel('lambda');
    %     title('Log Likelihood');
    % Initial values of parameters alpha, beta and lambda
    % [x,fval,exitflag,output] = fminsearch(f,x01,options);
    % fminsearch attempts to return a vector x that is a local minimizer of the
    % mathematical function near the starting vector
    % fminsearch uses the simplex search method of Lagarias et al.
    % [1] Lagarias, J. C., J. A. Reeds, M. H. Wright, and P. E. Wright.
    % “Convergence Properties of the Nelder-Mead Simplex Method in Low Dimensions.”
    % SIAM Journal of Optimization. Vol. 9, Number 1, 1998, pp. 112–147