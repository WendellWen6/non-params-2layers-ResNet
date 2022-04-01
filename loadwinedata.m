function [X_train,Y_train,X_test,Y_test] = loadwinedata(filename,flag)
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
        [~,noise,~,~] = generatenoise(d, 1, n);
        Y = [Y;noise];
        %Y = [Y;unifrnd(-0.1,0.1,d-1,n)];
    end


    % split to train, test sets

    split1 = round(n/5 *4);
    X_train = X(:,1:split1);
    X_test = X(:,split1+1:end);

    Y_train = Y(:,1:split1);
    Y_test = Y(:,split1+1:end);
end