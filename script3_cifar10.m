clear;
% running with full cifar-10 combined with alex

d = 10;
n_b = 334;

[X,Y,X_test,Y_test] = get_cifar10_data(d,50000);
A_final = zeros(d,d);
C_final = zeros(d,d);


for i = 1 : n_b
    n_start = 1 + 150 *(i - 1);
    n_end = 100 * i;
    if n_end > 50000
        n_end = 50000;
    end

    X_train = X(:,n_start:n_end);
    Y_train = Y(:,n_start:n_end);

    % lp
    C = relulp2_layer2(X_train, Y_train);
    B_lp = inv(C);
    H = C * Y_train - X_train;
    A_unscaled = relulp2_layer1(X_train, H);
    A_lp = rescale_layer1(X_train, H, A_unscaled);

    A_final = A_final + A_lp./n_b;
    C_final = C_final + C./n_b;

end

Y_pred_lp = C_final \ (max(A_final * X_test, 0) + X_test);
Y_accs_lp = calculate_acc(Y_pred_lp, Y_test);

