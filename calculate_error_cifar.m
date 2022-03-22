% function for calculating the err and accuracy for cifar-10 dataset under
% lp method
function [err,acc] = calculate_error_cifar(Y_pred,Y)
    acc = 0;
    [~,n] = size(Y_pred);
    for i = 1:n
        if Y_pred(i) >= Y(i) - 0.5 && Y_pred(i) < Y(i) + 0.5
            acc = acc + 1;
        end
    end
    err = mean(vecnorm(Y_pred - Y) ./ vecnorm(Y));
end