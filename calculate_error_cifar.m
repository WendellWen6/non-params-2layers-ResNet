% function for calculating the err and accuracy for cifar-10 dataset under
% lp method
function [err,acc] = calculate_error_cifar(Y_pred,Y)
    Y_pred_mean = round(mean(Y_pred));
    acc = 0;
    [d,n] = size(Y_pred_mean);
    for i = 1:n
        if Y_pred_mean(i) > 9
            Y_pred_mean(i) = 9;
        end
        if Y_pred_mean(i) <0
            Y_pred_mean(i) = 0;
        end
        if Y_pred_mean(i) >= Y(i) - 0.5 && Y_pred_mean(i) < Y(i) + 0.5
            acc = acc + 1;
        end
    end
    err = mean(vecnorm(Y_pred_mean - Y) ./ vecnorm(Y));
    
end