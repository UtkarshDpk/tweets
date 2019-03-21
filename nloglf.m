function [nloglf,grad] = nloglf(T,x)
%Function for computing negative Log-Likelihood function 
%and its gradient
k = length(T);
alpha = x(1);
beta = x(2);
lambda = x(3);
A = zeros(k,1);

sum1 = log(lambda + alpha.*A(1));
sum2 = exp(-beta.*(T(k)-T(1)))-1;

for i=2:k
    A(i) = exp(-beta.*(T(i)-T(i-1))).*(1+A(i-1));
    sum1 = sum1 + log(lambda + alpha.*A(i));
    sum2 = sum2 + exp(-beta.*(T(k)-T(i)))-1;
end

nloglf = lambda.*T(end) - (alpha./beta).*sum2 - sum1;

if nargout >1
    grad = zeros(3,1);
    sum3 = 1./lambda;
    sum4 = 0;
    sum5 = exp(-beta.*(T(k)-T(1)))-1;
    B = zeros(k,1);
    sum6 = 0;
    sum7 = (1./beta^2).*exp(-beta.*(T(k)-T(1)));
    sum8 = (1./beta).*(T(k)-T(1)).*exp(-beta.*(T(k)-T(1)));

    for i=2:k
        sum3 = sum3 + 1./(lambda + alpha.*A(i));
        sum4 = sum4 + A(i)./(lambda + alpha.*A(i));
        sum5 = sum5 + exp(-beta.*(T(k)-T(i)))-1;
        for j=1:i-1
           B(i) = (T(i)-T(j)).*(exp(-beta.*(T(i)-T(j))));
        end
        sum6 = sum6 + alpha.*B(i)./(lambda + alpha.*(A(i)));
        sum7 = sum7 + (1./beta^2).*exp(-beta.*(T(k)-T(i)));
        sum8 = sum8 + (1./beta).*(T(k)-T(i)).*exp(-beta.*(T(k)-T(i)));
    end

    grad(1) = - sum4 - sum5;
    grad(2) = alpha.*sum7 + alpha.*sum8 + sum6;
    grad(3) = T(k) - sum3;

% syms A(bta,a,b)
% syms sum1(lmbda,alfa,bta,a,b)
% syms sum2(alfa,bta,a,b)
% syms f(lmbda,a)
% syms g(alfa,bta,a,b)
% 
% A(bta,a,b) = 0;
% sum1(lmbda,alfa,bta,a,b) = log(lmbda + alfa.*A(bta,a,b));
% sum2(alfa,bta,a,b) = exp(-bta.*(subs(a,T(k))-subs(b,T(1))))-1;
% f(lmbda,a) = lmbda.*subs(a,T(end));
% g(alfa,bta,a,b) = (alfa./bta).*sum1(lmbda,alfa,bta,a,b);
% for i=2:k
%     A(bta,a,b) = exp(-bta.*(subs(a,T(i))-subs(b,T(i-1)))).*(1+(A(bta,a,b)));
%     sum1(lmbda,alfa,bta,a,b) = sum1(lmbda,alfa,bta,a,b) + log(lmbda + alfa.*A(bta,a,b));
%     sum2(alfa,bta,a,b) = sum2(alfa,bta,a,b) + (exp(-bta.*(subs(a,T(k))-subs(b,T(i))))-1);
% end
% nllf = -sum1(lmbda,alfa,bta,a,b) + f(lmbda,a) - g(alfa,bta,a,b);
% nloglf = matlabFunction(nllf);
end