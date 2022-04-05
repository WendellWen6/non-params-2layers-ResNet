% hist
h1 = histfit(Y_errs_lp);
hold on
h2 = histfit(Y_errs_bp);
h3 = histfit(Y_errs_qp);
set(get(get(h1(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h2(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h3(2),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
legend('LP','BP','QP');
xlabel('MSE of output error of quality of red wine dataset', 'Interpreter', 'latex');
ylabel('frequency', 'Interpreter', 'latex');
title('BP-LP-QP on wine datasets with ' + Pmethod + ' method on Y');