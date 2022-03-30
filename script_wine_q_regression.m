clear;
% running with red wine quality datasets

% set hyperparameters
trials = 2;
method = "repeat";
i = 1;

Y_errs_lp = zeros(trials, 1);
Y_errs_bp = zeros(trials, 1);
%Y_errs_qp = zeros(trials, 1);


for iterate = 1:trials

    %load data
    [X,Y,X_test,Y_test] = loadwinequality(method);
    %[X,Y,X_test,Y_test] = loadresidentialdata(method);
    
    % lp2
    %C = relulp2_layer2(X, Y);
    %B_lp = inv(C);
    %H = C * Y - X;
    %A_unscaled = relulp2_layer1(X, H);
    %A_lp = rescale_layer1(X, H, A_unscaled);   
    %Y_pred_lp = C \ (max(A_lp * X_test, 0) + X_test);
    

    % lp3
    C_lp = relulp3_layer2(X, Y);
    B_lp = inv(C_lp);
    H_lp = C_lp * Y - X;
    A_unscaled = relulp3_layer1(X, H_lp);
    A_lp = rescale_layer1(X, H_lp, A_unscaled);
    Y_pred_lp = C_lp \ (max(A_lp * X_test, 0) + X_test);

    
    % qp
    %[C_qp, H_qp] = reluqp2_layer2(X, Y);
    %B_qp = inv(C_qp);
    %A_unscaled = reluqp2_layer1(X, H_qp);
    %A_qp = rescale_layer1(X, H_qp, A_unscaled);
    %Y_pred_qp = C_qp \ (max(A_qp * X_test, 0) + X_test);
    

    

    
    % bp
    [A_bp, B_bp] = backprop2(X, Y, X_test, Y_test, 32, 1e-3, 1e-5, 256);
    Y_pred_bp = B_bp * (max(A_bp * X_test, 0) + X_test);
    



    % evaluations
    if method == "repeat"
        Y_pred_lp = mean(Y_pred_lp);
        Y_pred_bp = mean(Y_pred_bp);
        %Y_pred_qp = mean(Y_pred_qp);
        Y_test = round(mean(Y_test));

    elseif method == "non-repeat"
        Y_pred_lp = Y_pred_lp(1,:);
        Y_pred_bp = Y_pred_bp(1,:);
        %Y_pred_qp = Y_pred_qp(1,:);
        Y_test = round(mean(Y_test));
    end

    Y_errs_lp(i) = mean(vecnorm(Y_pred_lp - Y_test) ./ vecnorm(Y_test));
    Y_errs_bp(i) = mean(vecnorm(Y_pred_bp - Y_test) ./ vecnorm(Y_test));
    %Y_errs_qp(i) = mean(vecnorm(Y_pred_qp - Y_test) ./ vecnorm(Y_test));

    i = i + 1; 
end


h1 = histfit(Y_errs_lp);
hold on
h2 = histfit(Y_errs_bp);
%h3 = histfit(Y_errs_qp);
set(get(get(h1(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h2(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
%set(get(get(h3(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
legend('SGD','LP');
xlabel('empirical mean of output error of quality of red wine dataset $||\hat{\mathbf{y}} - \mathbf{y}|| / ||\mathbf{y}||$', 'Interpreter', 'latex');
ylabel('frequency', 'Interpreter', 'latex');
title('BP-LP on wine datasets with ' + method + ' method on Y');