% load with alexnet cifar10
load('cifar-10-batches-mat\data_batch_1.mat');
train = data;
train_label = labels;

load('cifar-10-batches-mat\data_batch_2.mat');
train = [train;data];
train_label = [train_label; labels];

load('cifar-10-batches-mat\data_batch_3.mat');
train = [train;data];
train_label = [train_label; labels];

load('cifar-10-batches-mat\data_batch_4.mat');
train = [train;data];
train_label = [train_label; labels];

load('cifar-10-batches-mat\data_batch_5.mat');
train = [train;data];
train_label = [train_label; labels];

X_train = train;
Y_train = train_label;

load('cifar-10-batches-mat\test_batch.mat');
X_test = data;
Y_test = labels;

net = alexnet;
layer = "fc6";
    
X_train = reshape(X_train', 32,32,3,[]);
X_train = permute(X_train, [2 1 3 4]);
X_train = imresize(X_train, [227,227]);
X_train = activations(net,X_train,layer,'OutputAs','rows');
X_train = double(normalize(X_train));

X_test = reshape(X_test', 32,32,3,[]);
X_test = permute(X_test, [2 1 3 4]);
X_test = imresize(X_test, [227,227]);
X_test = activations(net,X_test,layer,'OutputAs','rows');
X_test = double(normalize(X_test));

Y_train = double(Y_train);
Y_test = double(Y_test);

save("cifar10data2",'X_train','Y_train','X_test','Y_test');