clear;
% running with red wine quality datasets

% set hyperparameters
trials = 20;
i = 1;
lp3flag = false;
qpflag = false;

% Pmethod from ["addnoise", "addmodelnoise", "repeat", "repeatwithnoise"]
Pmethod = "repeatwithnoise";


% store the errors
Y_errs_lp = zeros(trials, 1);
Y_errs_bp = zeros(trials, 1);
Y_errs_qp = zeros(trials, 1);


for iterate = 1:trials

    % load data
    % could load other datasets by writing new load functions
    [X,Y,X_test,Y_test] = loadinsurance(Pmethod);
    

    if lp3flag
        % lp3
        C_lp = relulp3_layer2(X, Y);
        B_lp = inv(C_lp);
        H_lp = C_lp * Y - X;
        A_unscaled = relulp3_layer1(X, H_lp);
        A_lp = rescale_layer1(X, H_lp, A_unscaled);
        Y_pred_lp = C_lp \ (max(A_lp * X_test, 0) + X_test);
    else
        % lp2
        C = relulp2_layer2(X, Y);
        B_lp = inv(C);
        H = C * Y - X;
        A_unscaled = relulp2_layer1(X, H);
        A_lp = rescale_layer1(X, H, A_unscaled);   
        Y_pred_lp = C \ (max(A_lp * X_test, 0) + X_test);
    end

    

    if qpflag
        % qp
        [C_qp, H_qp] = reluqp2_layer2(X, Y);
        B_qp = inv(C_qp);
        A_unscaled = reluqp2_layer1(X, H_qp);
        A_qp = rescale_layer1(X, H_qp, A_unscaled);
        Y_pred_qp = C_qp \ (max(A_qp * X_test, 0) + X_test);
    end
    
    % bp
    [A_bp, B_bp] = backprop2(X, Y, X_test, Y_test, 16, 1e-3, 1e-5, 256);
    Y_pred_bp = B_bp * (max(A_bp * X_test, 0) + X_test);
    



    % evaluations
    if Pmethod == "repeat" || Pmethod == "repeatwithnoise"
        Y_pred_lp = mean(Y_pred_lp);
        Y_pred_bp = mean(Y_pred_bp);
        
        Y_test = round(mean(Y_test));

        if qpflag
            Y_pred_qp = mean(Y_pred_qp);
        end

    elseif Pmethod == "addnoise" || Pmethod == "addmodelnoise"
        Y_pred_lp = Y_pred_lp(1,:);
        Y_pred_bp = Y_pred_bp(1,:);
        Y_test = round(Y_test(1,:));

        if qpflag
            Y_pred_qp = Y_pred_qp(1,:);
        end
    end

    Y_errs_lp(i) = mymse(Y_test, Y_pred_lp); 
    Y_errs_bp(i) = mymse(Y_test, Y_pred_bp);

    if qpflag
        Y_errs_qp(i) = mymse(Y_test, Y_pred_qp);
    end



    i = i + 1; 
end

disp(mean(Y_errs_lp));
disp(mean(Y_errs_bp));
%disp(mean(Y_errs_qp));


h1 = histfit(Y_errs_lp);
hold on
h2 = histfit(Y_errs_bp);
%h3 = histfit(Y_errs_qp);
set(get(get(h1(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h2(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
%set(get(get(h3(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
legend('LP','BP');
xlabel('MSE of output error of quality of red wine dataset $||\hat{\mathbf{y}} - \mathbf{y}|| / ||\mathbf{y}||$', 'Interpreter', 'latex');
ylabel('frequency', 'Interpreter', 'latex');
title('BP-LP on wine datasets with ' + Pmethod + ' method on Y');