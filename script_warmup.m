% defining the hyperparameters
n = 128;
n_test = 20;
d = 10;

% initialize weights 
A = abs(randn(d,d));
B = randn(d,d);
% training set
X = randn(d,n);
Y = B * (max(A * X,0) + X);
% test set
X_test = randn(d,n_test);
Y_test = B * (max(A * X_test,0) + X_test);

% LP
C = relulp2_layer2(X, Y);
B_lp = inv(C);
H = C * Y - X;
A_unscaled = relulp2_layer1(X, H);
A_lp = rescale_layer1(X, H, A_unscaled);
Y_pred_lp = C \ (max(A_lp * X_test, 0) + X_test);

% QP
[C_qp, H_qp] = reluqp2_layer2(X, Y);
B_qp = inv(C_qp);
A_unscaled = reluqp2_layer1(X, H_qp);
A_qp = rescale_layer1(X, H_qp, A_unscaled);
Y_pred_qp = C_qp \ (max(A_qp * X_test, 0) + X_test);

% BP
[A_bp, B_bp] = backprop2(X, Y, X_test, Y_test, 32, 1e-3, 1e-5, 256);
Y_pred_bp = B_bp * (max(A_bp * X_test, 0) + X_test);

% Evaluation by L2
lp_error = mymse(Y_test, Y_pred_lp);
bp_error = mymse(Y_test, Y_pred_bp);
qp_error = mymse(Y_test, Y_pred_qp);

disp(lp_error);
disp(bp_error);
disp(qp_error);
