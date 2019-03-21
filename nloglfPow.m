function [ nloglf ] = nloglfPow(T,x)
% Function for computing negative Log-Likelihood function for Power Law

% sum = c0;
% sum2 = log(c0);
% for i=2:length(T)
%     sum = sum + c0*(((T(end)-T(i))./s0).^(-(1+p)));
%     sum2 = sum2 + log(c0*(((T(end)-T(i))./s0).^(-(1+p))));
% end
%
% sum3=0;
% for i=2:length(T)-1
%     for j=1:i
%         fun = @(t)(((t-T(j))./(t-T(1))).^(-(1+p)));
%         sum3 = sum3 + integral(fun,T(i),T(i+1));
%     end
% end
% intensity = lambda + sum;
% compensator = lambda*T(end) + c0*sum3;
% nloglf = -sum2 + compensator;
c0 = x(1);
s0 = x(2);
p = x(3);
lambda = x(4);
k = length(T);
A = zeros(k,1);
sum1 = log(lambda + c0.*A(1));
sum2 = ((T(k)-T(1))/s0)^(-(1+p));
for i=2:k
    %     s1 = s1 + log(k);
    %     for j=1:i
    %         s2 = s2 + (T(i)-T(j) + c).^(-(1+p));
    %     end
    %     s2 = (T(i)-T(i-1) + c).^(-(1+p))(1 + ;
    %     s3 = s3 + log(s2);
    if ((T(i)-T(i-1)) < s0)
        A(i) = 1;
    else
                 A(i) = ((T(i)-T(i-1))/s0 + A(i-1).^(-1/(1+p))).^(-(1+p))...
                     + ((T(i)-T(i-1))/s0).^(-(1+p));
%         for j = 1:i-1
%             A(i) = A(i) + ((T(i)-T(j))/s0).^(-(1+p));
%         end
    end
    sum1 = sum1 + log(lambda + c0.*A(i));
    if (T(k)-T(i) > s0)
        sum2 = sum2 + (T(k)-T(i))^(-p);
    end
    % end
    % s4 = 0;
    % for i=1:length(T)
    %     s4 = s4 + (1/(p*(s0^p)))-(((T(end)-T(i)+s0).^(-p))./p);
    % end
    %     nloglf = -s1 - s3 + c0*s4;
end
nloglf = lambda.*T(end) - (c0/p*(s0^(-(1+p)))).*sum2 - sum1 ;
end