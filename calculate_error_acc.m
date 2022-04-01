% function for calculating the err and accuracy

function [err,acc] = calculate_error_acc(method,Y_pred,Y)
    acc = 0;
    
    [~,n] = size(Y_pred);

    if method == "repeat"
        Y_pred = mean(Y_pred);
        Y = round(mean(Y));

        for i = 1:n
            if Y_pred(i) > 3
                Y_pred(i) = 3;
            elseif Y_pred(i) < 1
                Y_pred(i) = 1;
            end

            if round(Y_pred(i)) == Y(i)
                acc = acc + 1;
            end
        end

    elseif method == "onehot"
        Y_pred = softmax(Y_pred(1:3,:));
        [~, argmax] = max(Y_pred);
        Y_t = zeros(3,n);

        Y = Y(1:3,:);

        for i = 1:n
            Y_t(argmax(i),i) = 1;
            if Y_t(:,i) == Y(:,i)
                acc = acc + 1;
            end
        end
        

    elseif method == "non-repeat"
        Y_pred = Y_pred(1,:);
        Y = Y(1,:);

        for i = 1:n
            if Y_pred(i) > 3
                Y_pred(i) = 3;
            elseif Y_pred(i) < 1
                Y_pred(i) = 1;
            end
            if round(Y_pred(i)) == Y(i)
                acc = acc + 1;
            end
        end
        
    end

    err = mymse(Y,Y_pred);
    acc = acc / n;


end