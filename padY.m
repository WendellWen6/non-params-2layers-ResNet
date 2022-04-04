% padding Y methods
function Y_pad =  padY(Y,method,d)
    [t,n] = size(Y);

    % method1: padding with random distribution noise
    if method == "addnoise"
        Y_pad = [Y;unifrnd(-0.1,0.1,d-t,n)];
    
    % method2: padding with random generated data from our model
    elseif method == "addmodelnoise"
        Y_pad = [Y;generatenoise(d, t, n)];
    
    % method3: duplicating without adding noise
    elseif method == "repeat"
        Y_pad = repmat(Y,d,1);
    
    % method4: duplicating with random distribution noise
    elseif method == "repeatwithnoise"
        Y_pad = repmat(Y,d,1);
        Y_pad = Y_pad + unifrnd(-0.1,0.1,d,n);

    % mthoed5: PCA, then padding with random distribution noise
    elseif method == "onehot"
        Y_pad = categorical(Y);
        Y_pad = onehotencode(Y_pad,1);
        [t,n] = size(Y_pad);
        Y_pad = [Y_pad;unifrnd(-0.1,0.1,d-t,n)];
    end
end