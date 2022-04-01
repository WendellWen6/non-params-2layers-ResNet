% function for calculating accuracy for cifar10
function acc = calculate_acc(Y_pred,Y)
    acc = 0;
    [~,n] = size(Y_pred);


    Y_pred = softmax(Y_pred(1:10,:));
    [~, argmax] = max(Y_pred);
    Y_t = zeros(10,n);

    Y = Y(1:10,:);

    for i = 1:n
        Y_t(argmax(i),i) = 1;
        if Y_t(:,i) == Y(:,i)
            acc = acc + 1;
        end
    end
    acc = acc / n * 100;
end