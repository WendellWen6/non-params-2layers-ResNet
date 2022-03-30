function error = mymse(Y,Y_pred)
    [d,~] = size(Y);
    if d == 1
        error = mean(square(Y - Y_pred));
    else
        error = mean(sum(square(Y - Y_pred)));
    end
end