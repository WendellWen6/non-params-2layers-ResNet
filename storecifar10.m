% store the cifar-10 dataset and slipt into train,val,test sets
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

X_train = normalize(double(train));
Y_train = double(train_label);

load('C:\Users\13945\Desktop\honour\code\non-params-2layers-ResNet\cifar-10-batches-mat\test_batch.mat');
X_test = normalize(double(data));
Y_test = double(labels);


save("cifar10data",'X_train','Y_train','X_test','Y_test');
