function [X_train,Y_train,X_test,Y_test] = loadinsurance(method)
    % load data
    data = readtable("insurance\insurance.csv");
    data = data{:,:};  

    % randomrize the data
    data = data(randperm(size(data,1)),:);

    % PCA categorical features
    X = data(:,1:5);
    region = data(:,6);
    region1 = categorical(region);
    region2 = onehotencode(region1,2);
    X = [X,region2];

    % get X, normalize the X by each column (feature)
    X = normalize(X).';
    [d, n] = size(X);

    % get Y
    Y = data(:,end).';
    
    % padding Y by different methods
    Y = padY(Y,method,d);

    % split to train, test sets

    split1 = round(n/5 *4) + 2;

    X_train = X(:,1:split1);
    X_test = X(:,split1+1:end);

    Y_train = Y(:,1:split1);
    Y_test = Y(:,split1+1:end);
end