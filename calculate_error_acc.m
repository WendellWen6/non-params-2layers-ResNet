% function for calculating the err and accuracy

function [err,acc] = calculate_error_acc(method,Y_pred,Y)
    acc = 0;
    
    [~,n] = size(Y);

    if method == "repeat"
        Y_pred = mean(Y_pred);
        Y = round(mean(Y));

        for i = 1:n
            if round(Y_pred(i)) == Y(i)
                acc = acc + 1;
            end
        end

    elseif method == "onehot"
        Y_pred = Y_pred(1:3,:);
        Y = Y(1:3,:);

        for i = 1:n
            if round(Y_pred(:,i)) == Y(:,i)
                acc = acc + 1;
            end
        end
        

    elseif method == "non-repeat"
        Y_pred = Y_pred(1,:);
        Y = Y(1,:);

        for i = 1:n
            if round(Y_pred(i)) == Y(i)
                acc = acc + 1;
            end
        end
        
    end

    err = mean(vecnorm(Y_pred - Y) ./ vecnorm(Y));
    acc = acc / n;


end