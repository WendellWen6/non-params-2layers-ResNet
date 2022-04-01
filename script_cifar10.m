clear;
% running with cifar-10



Y_accs_2d_lp = zeros(10, 20);
%Y_accs_2d_qp = zeros(10, 20);

x_axis = zeros(10, 1);
y_axis = zeros(10, 1);

% change dimension number
for i = 1 : 10
    %load data
    d = 10 + 3 * (i - 1);
    y_axis(i) = d ;
    [X,Y,X_test,Y_test] = get_cifar10_data(d,50000);
    
    % change training sample number
    for j = 1 : 20
        n = 100 + 50 * (j-1);
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
        
        % qp
        %[C_qp, H_qp] = reluqp2_layer2(X_train, Y_train);
        %B_qp = inv(C_qp);
        %A_unscaled = reluqp2_layer1(X_train, H_qp);
        %A_qp = rescale_layer1(X_train, H_qp, A_unscaled);
        %Y_pred_qp = C_qp \ (max(A_qp * X_test, 0) + X_test);

        Y_accs_2d_lp(i,j) = calculate_acc(Y_pred_lp, Y_test);
        %Y_accs_2d_qp(i,j) = calculate_acc(Y_pred_qp, Y_test);
    end

end

data = Y_accs_2d_lp;

%heatmap(x_axis, y_axis, Y_errs_2d_lp);
%heatmap(x_axis, y_axis, Y_errs_2d_bp);
%// Define integer grid of coordinates for the above data
[H,V] = meshgrid(1:size(data,2), 1:size(data,1));

%// Define a finer grid of points
[H2,V2] = meshgrid(1:0.01:size(data,2), 1:0.01:size(data,1));

%// Interpolate the data and show the output
outData = interp2(H, V, data, H2, V2, 'linear');
imagesc(outData);

%// Cosmetic changes for the axes
set(gca, 'XTick', linspace(1,size(H2,2),size(H,2))); 
set(gca, 'YTick', linspace(1,size(H2,1),size(H,1)));
set(gca, 'XTickLabel', x_axis);
set(gca, 'YTickLabel', y_axis);

%// Add colour bar

xlabel('sample size $n$', 'Interpreter', 'latex');
ylabel('number of dimensions $d$', 'Interpreter', 'latex');
title('LP');

hc3 = colorbar;
set(get(hc3,'label'),'string','Accuracy in % over full test set');






