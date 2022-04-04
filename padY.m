% padding Y methods
% method1: padding with random distribution noise
% method2: padding with random generated data from our model
% method3: duplicating without adding noise
% method4: duplicating with random distribution noise
% mthoed5: PCA, then padding with random distribution noise
function Y_pad =  padY(Y,method,d)
    [t,n] = size(Y);

    if method == "addnoise"
        Y_pad = [Y;unifrnd(-0.1,0.1,d-t,n)];

    elseif method == "addmodelnoise"
        Y_pad = [Y;generatenoise(d, t, n)];

    elseif method == "repeat"
        Y_pad = repmat(Y,d,1);
    
    elseif method == "repeatwithnoise"
        Y_pad = repmat(Y,d,1);
        Y_pad = Y_pad + unifrnd(-0.1,0.1,d,n);
    
    elseif method == "onehot"
        Y_pad = categorical(Y);
        Y_pad = onehotencode(Y_pad,1);
        [t,n] = size(Y_pad);
        Y_pad = [Y_pad;unifrnd(-0.1,0.1,d-t,n)];
    end
end