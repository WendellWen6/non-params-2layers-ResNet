function [X,Y,X_val,Y_val] = get_cifar10_data(d,n)
    % preprocessing the datasets
    load("cifar10data",'X_train','Y_train','X_val','Y_val');
    
    % do pca on raw train data
    [~, X_train_pca] = pca(X_train, 'NumComponents',d);
    % transpose the X
    X = X_train_pca';
    X = X(:,1:n);
    
    % transpose the Y
    Y = Y_train';
    Y = Y(:,1:n);
    % padding Y by duplicating and adding noise
    %Y = repmat(Y,d,1);
    %Y = Y + unifrnd(-0.1,0.1,d,n);
    % padding without duplicating
    Y = [Y;unifrnd(-0.1,0.1,d-1,n)];
        
    % do pca on raw validation data
    [~, X_val_pca] = pca(X_val, 'NumComponents',d);
    X_val = X_val_pca';
    Y_val = Y_val';
    
end