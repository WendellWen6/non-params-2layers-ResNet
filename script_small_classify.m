clear;
% running with wine datasets

% set hyperparameters
filename = "winedatasets\wine.data";
trials = 10;
method = "repeat";
batchsize = 2;
i = 1;

Y_errs_lp = zeros(trials, 1);
Y_errs_bp = zeros(trials, 1);
accs_lp = zeros(trials, 1);
accs_bp = zeros(trials, 1);


for iterate = 1:trials

    %load data
    [X,Y,X_val,Y_val,X_test,Y_test] = loadwinedata(filename, method);
    
    % lp
    C = relulp2_layer2(X, Y);
    B_lp = inv(C);
    H = C * Y - X;
    A_unscaled = relulp2_layer1(X, H);
    A_lp = rescale_layer1(X, H, A_unscaled);   
    Y_pred_lp = C \ (max(A_lp * X_val, 0) + X_val);
    
    
    
    % bp
    [A_bp, B_bp] = backprop2(X, Y, X_val, Y_val, batchsize, 1e-3, 1e-5, 256);
    Y_pred_bp = B_bp * (max(A_bp * X_val, 0) + X_val);
    
    % evaluations
    
    [Y_errs_lp(i), accs_lp(i)] = calculate_error_acc(method,Y_pred_lp,Y_val);
    [Y_errs_bp(i), accs_bp(i)] = calculate_error_acc(method,Y_pred_bp,Y_val);

    i = i + 1; 
end


h1 = histfit(accs_bp);
hold on
h2 = histfit(accs_lp);
set(get(get(h1(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h2(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
legend('SGD','Ours');
xlabel('Accuracy of classification $||\hat{\mathbf{y}} - \mathbf{y}|| / ||\mathbf{y}||$', 'Interpreter', 'latex');
ylabel('frequency', 'Interpreter', 'latex');
title('BP-LP on wine datasets with ' + method + ' method on Y');






