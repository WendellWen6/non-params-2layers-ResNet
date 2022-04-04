function [] = expmain()
    % one-hot encoding the label
    if method == "onehot"
        Y = categorical(Y);
        Y = onehotencode(Y,1);
        Y = [Y;unifrnd(-0.1,0.1,d-3,n)];
    % addding noise with duplicating   
    elseif method == "repeat"
        Y = repmat(Y,d,1);
        Y = Y + unifrnd(-0.1,0.1,d,n);

    % addding noise without duplicating    
    elseif method == "non-repeat"
        [~,noise,~,~] = generatenoise(d, 1, n);
        Y = [Y;noise];
        %Y = [Y;unifrnd(-0.1,0.1,d-1,n)];
    end
end