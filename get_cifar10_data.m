function [X,Y,X_test,Y_test] = get_cifar10_data(d,n)
    % preprocessing the datasets
    load("cifar10data",'X_train','Y_train','X_test','Y_test');
    
    % do pca on raw train data
    [~, X_train_pca] = pca(X_train, 'NumComponents',d);
    % transpose the X
    X = X_train_pca';
    X = X(:,1:n);
    
    % transpose the Y
    Y = Y_train';
    Y = Y(:,1:n);
    Y = categorical(Y);
    Y = onehotencode(Y,1);

    if d > 10
        % padding without duplicating
        Y = [Y;unifrnd(-0.1,0.1,d-10,n)];      
    end
        
    % do pca on raw validation data
    [~, X_test_pca] = pca(X_test, 'NumComponents',d);
    X_test = X_test_pca';
    Y_test = Y_test';
    Y_test = categorical(Y_test);
    Y_test = onehotencode(Y_test,1);
    
end