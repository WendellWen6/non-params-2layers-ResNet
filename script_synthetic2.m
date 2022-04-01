clear;

n_test = 100;
t = 1;

Y_errs_2d_lp = zeros(8, 8);
Y_errs_2d_bp = zeros(8, 8);

x_axis = zeros(8, 1);
y_axis = zeros(8, 1);


for i = 1 : 8
  d = 6 + i * 2;
  y_axis(i) = d;

  A = abs(randn(d,d));
  B = randn(d,d);

  % test sets
  X_test = randn(d,n_test);
  Y_test = B * (max(A * X_test,0) + X_test);
  Y_test = Y_test(1:t,:);

  for j = 1 : 8
    n = 128 + j * 32;
    x_axis(j) = n;
    
    Y_errs_lp = zeros(16, 1);
    Y_errs_bp = zeros(16, 1);
    T = 1;
    while T <= 2
      % training set
      X = randn(d,n);
      Y = B * (max(A * X,0) + X);
      Y = Y(1:t,:);

      % repeat the target row
      Y = repmat(Y,d,1);
      
      
      % lp3
      C_lp = relulp3_layer2(X, Y);
      B_lp = inv(C_lp);
      H_lp = C_lp * Y - X;
      A_unscaled = relulp3_layer1(X, H_lp);
      A_lp = rescale_layer1(X, H_lp, A_unscaled);
      Y_pred_lp = C_lp \ (max(A_lp * X_test, 0) + X_test);
      
      % bp
      [A_bp, B_bp] = backprop2(X, Y, X_test, Y_test, 32, 1e-3, 1e-5, 256);
      Y_pred_bp = B_bp * (max(A_bp * X_test, 0) + X_test);
      

      % for repeat method
      Y_errs_lp(T) = mymse(Y_test, mean(Y_pred_lp));
      Y_errs_bp(T) = mymse(Y_test, mean(Y_pred_bp));
      T = T + 1;
    end

    Y_errs_2d_lp(i, j) = mean(Y_errs_lp);
    Y_errs_2d_bp(i, j) = mean(Y_errs_bp);
    
  end
end

