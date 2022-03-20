% running with cifar-10
load('C:\Users\13945\Desktop\honour\code\non-params-2layers-ResNet\cifar-10-batches-mat\data_batch_1.mat');
train = data;
train_label = labels;

load('C:\Users\13945\Desktop\honour\code\non-params-2layers-ResNet\cifar-10-batches-mat\data_batch_2.mat');
train = [train;data];
train_label = [train_label; labels];

load('C:\Users\13945\Desktop\honour\code\non-params-2layers-ResNet\cifar-10-batches-mat\data_batch_3.mat');
train = [train;data];
train_label = [train_label; labels];

load('C:\Users\13945\Desktop\honour\code\non-params-2layers-ResNet\cifar-10-batches-mat\data_batch_4.mat');
train = [train;data];
train_label = [train_label; labels];

load('C:\Users\13945\Desktop\honour\code\non-params-2layers-ResNet\cifar-10-batches-mat\data_batch_5.mat');
train = [train;data];
train_label = [train_label; labels];

load('C:\Users\13945\Desktop\honour\code\non-params-2layers-ResNet\cifar-10-batches-mat\test_batch.mat');
test = data;
test_label = labels;


%pre-process the data
% X in (d, N)
% normalize X in to N(0,1)
% Y in (d, N)
% transfer Y into square matrix by duplicating d times

X_train = normalize(double(transpose(train)));
Y_train = double(transpose(train_label));


X_st = X_train(:,1:50);
Y_st = Y_train(:,1:50);
Y_st = repmat(Y_st,3072,1);

X_test = normalize(double(transpose(test)));
Y_test = double(transpose(test_label));

X_stest = X_test(:,1:10);
Y_stest = Y_test(:,1:10);
Y_stest = repmat(Y_stest,3072,1);

%adding random noise to Y

save("cifar10data",'X_train','Y_train','X_test','Y_test','X_st','Y_st','X_stest','Y_stest');
