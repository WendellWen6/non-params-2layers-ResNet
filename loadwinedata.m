function [X_train,Y_train,X_val,Y_val,X_test,Y_test] = loadwinedata(filename,flag)
    data = importdata(filename);
    
    % randomrize the data
    data = data(randperm(size(data,1)),:);
    
    % normalize the X by each column (feature)
    X = normalize(data(:,2:end))';
    [d, n] = size(X);

    Y = data(:,1)';

    % one-hot encoding the label
    if flag == "onehot"
        Y = categorical(Y);
        Y = onehotencode(Y,1);
        Y = [Y;unifrnd(-0.1,0.1,d-3,n)];
    % addding noise with duplicating   
    elseif flag == "repeat"
        Y = repmat(Y,d,1);
        Y = Y + unifrnd(-0.1,0.1,d,n);

    % addding noise without duplicating    
    elseif flag == "non-repeat"
        Y = [Y;unifrnd(-0.1,0.1,d-1,n)];
    end


    % split to train, val, test sets

    split1 = round(n/5*4);
    split2 = round(n/10 *9);
    X_train = X(:,1:split1);
    X_val = X(:,split1+1:split2);
    X_test = X(:,split2+1:end);

    Y_train = Y(:,1:split1);
    Y_val = Y(:,split1+1:split2);
    Y_test = Y(:,split2+1:end);
end