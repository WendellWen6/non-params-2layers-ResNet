clear;
% running with wine datasets

% set hyperparameters
trials = 20;
i = 1;

% Pmethod from ["addnoise", "addmodelnoise", "repeat", "repeatwithnoise","onehot"]
Pmethod = "repeatwithnoise";

Y_errs_lp = zeros(trials, 1);
Y_errs_bp = zeros(trials, 1);
Y_errs_qp = zeros(trials, 1);
accs_lp = zeros(trials, 1);
accs_bp = zeros(trials, 1);
accs_qp = zeros(trials, 1);

for iterate = 1:trials

    % load data
    % could load other datasets by writing new load functions
    [X,Y,X_test,Y_test] = loadwinedata(Pmethod);
    
    % lp
    C = relulp2_layer2(X, Y);
    B_lp = inv(C);
    H = C * Y - X;
    A_unscaled = relulp2_layer1(X, H);
    A_lp = rescale_layer1(X, H, A_unscaled);   
    Y_pred_lp = C \ (max(A_lp * X_test, 0) + X_test);
    
    % qp
    [C_qp, H_qp] = reluqp2_layer2(X, Y);
    B_qp = inv(C_qp);
    A_unscaled = reluqp2_layer1(X, H_qp);
    A_qp = rescale_layer1(X, H_qp, A_unscaled);
    Y_pred_qp = C_qp \ (max(A_qp * X_test, 0) + X_test);
    
    % bp
    [A_bp, B_bp] = backprop2(X, Y, X_test, Y_test, 2, 1e-3, 1e-5, 256);
    Y_pred_bp = B_bp * (max(A_bp * X_test, 0) + X_test);
    
    % evaluations
    
    [Y_errs_lp(i), accs_lp(i)] = calculate_error_acc(Pmethod,Y_pred_lp,Y_test);
    [Y_errs_bp(i), accs_bp(i)] = calculate_error_acc(Pmethod,Y_pred_bp,Y_test);
    [Y_errs_qp(i), accs_qp(i)] = calculate_error_acc(Pmethod,Y_pred_qp,Y_test);

    i = i + 1; 
end


h1 = histfit(accs_bp);
hold on
h2 = histfit(accs_lp);
h3 = histfit(accs_qp);
set(get(get(h1(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h2(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h3(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
legend('BP','LP','QP');
xlabel('Accuracy of classification', 'Interpreter', 'latex');
ylabel('frequency', 'Interpreter', 'latex');
title('BP-LP-QP on wine datasets with ' + Pmethod + ' method on Y');

disp(mean(accs_lp));
disp(mean(accs_bp));
disp(mean(accs_qp));

disp(std(accs_lp));
disp(std(accs_bp));
disp(std(accs_qp));



