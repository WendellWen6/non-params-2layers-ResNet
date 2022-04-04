function [X_train,Y_train,X_test,Y_test] = loadwinedata(method)
    data = importdata("winedatasets\wine.data");
    
    % randomrize the data
    data = data(randperm(size(data,1)),:);
    
    % normalize the X by each column (feature)
    X = normalize(data(:,2:end))';
    [d, n] = size(X);

    Y = data(:,1)';
    
    Y = padY(Y,method,d);


    % split to train, test sets

    split1 = round(n/5 *4);
    X_train = X(:,1:split1);
    X_test = X(:,split1+1:end);

    Y_train = Y(:,1:split1);
    Y_test = Y(:,split1+1:end);
end