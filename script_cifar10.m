% running with cifar-10
%load data
%load("cifar10data",'X_train','Y_train','X_test','Y_test');

%smaller size
load('cifar10data','X_st','Y_st','X_stest','Y_stest');


%set hyperparams for trainning
T_gt = 1;
trials = 1;

%[d, n] = size(X_train);
[d, n] = size(X_st);

A_errs_lp = zeros(T_gt * trials, 1);
B_errs_lp = zeros(T_gt * trials, 1);
Y_errs_lp = zeros(T_gt * trials, 1);

A_errs_bp = zeros(T_gt * trials, 1);
B_errs_bp = zeros(T_gt * trials, 1);
Y_errs_bp = zeros(T_gt * trials, 1);

i = 1;

for T_gt_it = 1 : T_gt
  [A_g, B_g] = params_gen_res_relu(500, d);
  for trails_it = 1 : trials
    
    % lp
    C = relulp2_layer2(X_st, Y_st);
    disp(size(C));
    B_lp = inv(C);
    H = C * Y_st - X_st;
    A_unscaled = relulp2_layer1(X_st, H);
    A_lp = rescale_layer1(X_train, H, A_unscaled);
    Y_pred_lp = C \ (max(A_lp * X_stest, 0) + X_stest);
    
    A_errs_lp(i) = norm(A_lp - A_g) / norm(A_g);
    B_errs_lp(i) = norm(B_lp - B_g) / norm(B_g);
    Y_errs_lp(i) = mean(vecnorm(Y_pred_lp - Y_stest) ./ vecnorm(Y_stest));
    
    % bp
%     [A_bp, B_bp] = backprop2(X_train, Y_train, X_test, Y_test, 32, 1e-3, 1e-5, 256);
%     Y_pred_bp = B_bp * (max(A_bp * X_test, 0) + X_test);
%     
%     A_errs_bp(i) = norm(A_bp - A_g) / norm(A_g);
%     B_errs_bp(i) = norm(B_bp - B_g) / norm(B_g);
%     Y_errs_bp(i) = mean(vecnorm(Y_pred_bp - Y_test) ./ vecnorm(Y_test));
    
    i = i + 1;
  end
end

%h1 = histfit(Y_errs_bp);
hold on
h2 = histfit(Y_errs_lp);
%set(get(get(h1(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h2(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
legend('SGD','Ours');
xlabel('empirical mean of output error $||\hat{\mathbf{y}} - \mathbf{y}|| / ||\mathbf{y}||$', 'Interpreter', 'latex');
ylabel('frequency', 'Interpreter', 'latex');
title('BP-LP-noiseless');





