function [X_train,Y_train,X_test,Y_test] = loadwinequality(method)
    % load data
    data = readtable("winequality\winequality-red.csv");
    data = data{:,:};

    % randomrize the data
    data = data(randperm(size(data,1)),:);

    % get X, normalize the X by each column (feature)
    X = normalize(data(:,1:end-1)).';
    [d, n] = size(X);

    % get Y
    Y = data(:,end).';
    
    % padding Y by different methods
    Y = padY(Y,method,d);

    % split to train, test sets

    split1 = round(n/5 *4) + 1;

    X_train = X(:,1:split1);
    X_test = X(:,split1+1:end);

    Y_train = Y(:,1:split1);
    Y_test = Y(:,split1+1:end);
end