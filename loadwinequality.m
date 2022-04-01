function [X_train,Y_train,X_test,Y_test] = loadwinequality(flag)
    % load data
    data = readtable("winequality\winequality-red.csv");
    data = data{:,:};
    % randomrize the data
    data = data(randperm(size(data,1)),:);

    % get X, normalize the X by each column (feature)
    X = normalize(data(:,1:end-1))';
    [d, n] = size(X);

    % get Y
    Y = data(:,end)';

    % addding noise with duplicating   
    if flag == "repeat"
        Y = repmat(Y,d,1);
        Y = Y + unifrnd(-0.1,0.1,d,n);

    % addding noise without duplicating    
    elseif flag == "non-repeat"
        %Y = [Y;unifrnd(-0.1,0.1,d-1,n)];
        Y = [Y;generatenoise(d, 1, n)];
    end

    % split to train, test sets

    split1 = round(n/5 *4) + 1;

    X_train = X(:,1:split1);
    X_test = X(:,split1+1:end);

    Y_train = Y(:,1:split1);
    Y_test = Y(:,split1+1:end);
    

end