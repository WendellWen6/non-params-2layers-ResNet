function error = mymse(Y,Y_pred)
    [d,~] = size(Y);
    if d == 1
        error = mean((Y - Y_pred).^2);
    else
        error = mean(sum((Y - Y_pred).^2));
    end
end