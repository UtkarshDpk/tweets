function [xmin, fval, history] = runfmincon(objfun,x0,lowerBound,A,b)
% Set up shared variables with OUTFUN
history.x = [];
history.fval = [];
%searchdir = [];
opts = optimoptions(@fmincon,'Algorithm','interior-point','SubproblemAlgorithm', 'cg','FunValCheck', 'on'...
    ,'Display', 'iter-detailed','Diagnostics', 'on',...%'MaxIter',10,...%'SpecifyObjectiveGradient',true,
    'HessianApproximation','bfgs','TolFun',sqrt(eps),'TolX',sqrt(eps),'OutputFcn',@plotLLAtIter);
[xsol,fval] = fmincon(objfun,x0,A,b,[],[],lowerBound,[],[],opts);
    function stop = plotLLAtIter(x,optimValues,state)
        % Plots Log Likelihood func after each iteration
        stop = false;
        switch state
%             case 'init'
%                 hold on
            case 'iter'
                % Concatenate current point and objective function
                % value with history. x must be a row vector.
                history.fval = [history.fval; optimValues.fval];
                history.x = [history.x; x];
%             case 'done'
%                 hold off
            otherwise
        end
        
    end
xmin = xsol;
end

