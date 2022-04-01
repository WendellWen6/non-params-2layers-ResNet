clear;
% running with cifar-10



Y_accs_2d_lp = zeros(10, 20);
x_axis = zeros(10, 1);
y_axis = zeros(10, 1);

% change dimension number
for i = 1 : 1
    %load data
    d = 10 + i - 1;
    y_axis(i) = d ;
    [X,Y,X_test,Y_test] = get_cifar10_data(d,50000);
    
    % change training sample number
    for j = 1 : 20
        n = 100 + 20 * (j-1);
        x_axis(j) = n;
        X_train = X(:,1:n);
        Y_train = Y(:,1:n);
        % lp
        C = relulp2_layer2(X_train, Y_train);
        B_lp = inv(C);
        H = C * Y_train - X_train;
        A_unscaled = relulp2_layer1(X_train, H);
        A_lp = rescale_layer1(X_train, H, A_unscaled);
            
        Y_pred_lp = C \ (max(A_lp * X_test, 0) + X_test);
        
        Y_accs_2d_lp(i,j) = calculate_acc(Y_pred_lp, Y_test);
    end

end








