clear;
% generate one dimension target data
n = 128;
n_test = 100;
d = 10;
t = 1;
i = 1;
trials = 10;

Y_errs_lp = zeros(trials, 1);
Y_errs_bp = zeros(trials, 1);
Y_errs_qp = zeros(trials, 1);


for iterate = 1:trials
    % generate training and test sets

    A = abs(randn(d,d));
    B = randn(d,d);
    
    X = randn(d,n);
    Y = B * (max(A * X,0) + X);
    Y = Y(1:t,:);

    % padding noise follow same function but different weights
    %[~,noise,~,~] = generatenoise(d,t,n);
    %Y = [Y;noise];

    % padding uniform noise
    %Y = [Y;unifrnd(-0.1,0.1,d-t,n)];

    % repeat the target row
    Y = repmat(Y,d,1);

    % with noise or not
    Y = Y + unifrnd(-0.1,0.1,d,n);
    

    X_test = randn(d,n_test);
    Y_test = B * (max(A * X_test,0) + X_test);
    Y_test = Y_test(1:t,:);
    
    
    % lp3
    C_lp = relulp3_layer2(X, Y);
    B_lp = inv(C_lp);
    H_lp = C_lp * Y - X;
    A_unscaled = relulp3_layer1(X, H_lp);
    A_lp = rescale_layer1(X, H_lp, A_unscaled);
    Y_pred_lp = C_lp \ (max(A_lp * X_test, 0) + X_test);
    
    % QP
    [C_qp, H_qp] = reluqp2_layer2(X, Y);
    B_qp = inv(C_qp);
    A_unscaled = reluqp2_layer1(X, H_qp);
    A_qp = rescale_layer1(X, H_qp, A_unscaled);
    Y_pred_qp = C_qp \ (max(A_qp * X_test, 0) + X_test);
    
    % BP
    [A_bp, B_bp] = backprop2(X, Y, X_test, Y_test, 8, 1e-3, 1e-5, 256);
    Y_pred_bp = B_bp * (max(A_bp * X_test, 0) + X_test);
    
    % non-repeat method
    %lp_error = mymse(Y_test, Y_pred_lp(1,:));
    %bp_error = mymse(Y_test, Y_pred_bp(1,:));
    %qp_error = mymse(Y_test, Y_pred_qp(1,:));
    
    % for repeat method
    Y_errs_lp(i) = mymse(Y_test, mean(Y_pred_lp));
    Y_errs_bp(i) = mymse(Y_test, mean(Y_pred_bp));
    Y_errs_qp(i) = mymse(Y_test, mean(Y_pred_qp));
    i = i + 1;
end

h1 = histfit(Y_errs_lp);
hold on
h2 = histfit(Y_errs_bp);
h3 = histfit(Y_errs_qp);
set(get(get(h1(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h2(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h3(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
legend('LP','SGD','QP');
xlabel('Mean square errors of LP-BP-QP ', 'Interpreter', 'latex');
ylabel('frequency', 'Interpreter', 'latex');
title('LP-BP-QP on synthetic datasets with duplicating method on Y');
