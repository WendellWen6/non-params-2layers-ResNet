clear;
% running with cifar-10

%load data
[X,Y,X_val,Y_val] = get_cifar10_data(10,100);

% lp
C = relulp2_layer2(X, Y);
B_lp = inv(C);
H = C * Y - X;
A_unscaled = relulp2_layer1(X, H);
A_lp = rescale_layer1(X, H, A_unscaled);
    
Y_pred_lp = C \ (max(A_lp * X_val, 0) + X_val);

[Y_errs_lp,acc_lp] = calculate_error_acc(Y_pred_lp(1,:),Y_val);
% bp
%[A_bp, B_bp] = backprop2(X, Y, X_val, Y_val, 40, 1e-3, 1e-5, 256);
%Y_pred_bp = mean(B_bp * (max(A_bp * X_val, 0) + X_val));
     
%Y_errs_bp = mean(vecnorm(Y_pred_bp - Y_val) ./ vecnorm(Y_val));








