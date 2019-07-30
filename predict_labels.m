function [pred_labels]=predict_labels(train_inputs,train_labels,test_inputs)
%% Load and Processing data
% load features

XTrain = train_inputs;
XTest  = test_inputs;
YTrain = train_labels;

Total = [XTrain; XTest];
NTrain = size(XTrain, 1);
XTweet = Total(:, 22:end);
XDemo = Total(:, 1:21);

[~, score] = pca(XTweet);
XTweetPCA = score(:, 1:100);

XTrains = [XDemo((1:NTrain),:), XTweetPCA((1:NTrain), :)];
ZTrains = zscore(XTrains);

XTests = [XDemo((NTrain + 1): end, :), XTweetPCA((NTrain + 1) : end, :)];
ZTests = zscore(XTests);


%% LR
LR_Y_est = zeros(size(XTrain, 1), 9);
LR_Y_pred = zeros(size(XTest, 1), 9);

for n = 1:9
    Mdl = fitrlinear(ZTrains, YTrain(:,n),'Regularization', 'lasso', 'Solver', 'sparsa');
    LR_Y_est(:, n) = predict(Mdl, ZTrains);
    LR_Y_pred(:, n) = predict(Mdl, ZTests);
   if n ~= 3 && n ~= 8
       LR_Y_est(:, n) = round(LR_Y_est(:, n));
       LR_Y_pred(:, n) = round(LR_Y_pred(:, n));
   end
end


ZPCA = zscore([XTweetPCA]);
ZTrains2 = ZPCA((1:NTrain), :);
ZTests2 = ZPCA((NTrain+1) : end, :);

%% KNN
KNN_Y_est = zeros(size(XTrain, 1), 9);
KNN_Y_pred = zeros(size(XTest, 1), 9);

for n = 1:9
    KNN_Y_est(:,n) = k_nearest_neighbours(ZTrains2, YTrain(:,n), ZTrains2, 20, 'l2');
    KNN_Y_pred(:,n) = k_nearest_neighbours(ZTrains2,YTrain(:,n), ZTests2, 20,'l2');
end

%% NN
NN_Y_est = zeros(size(XTrain, 1), 9);
NN_Y_pred = zeros(size(XTest, 1), 9);

for n = 1:9
    net = feedforwardnet(25);
    net.layers{1}.transferFcn = 'poslin';
    net.performFcn = 'mse' ;    
    net.performParam.regularization = 0.1;
    net = trainlm(net,ZTrains2',YTrain(:,n)');
    NN_Y_est(:,n) = net(ZTrains2');
    NN_Y_pred(:,n)= net(ZTests2');
end

%% SVM
SVM_Y_est = zeros(size(XTrain, 1), 9);
SVM_Y_pred = zeros(size(XTest, 1), 9);

for n = 1:9
    Mdl = fitrsvm(ZTrains, YTrain(:, n));
    SVM_Y_est(:,n) = predict(Mdl, ZTrains);
    SVM_Y_pred(:,n) = predict(Mdl, ZTests);
   if n ~= 3 && n ~= 8
       SVM_Y_est(:,n) = round(SVM_Y_est(:, n));
        SVM_Y_pred(:,n) = round(SVM_Y_pred(:, n)); 
   end
end

%% RF
RF_Y_est = zeros(size(XTrain, 1), 9);
RF_Y_pred = zeros(size(XTest, 1), 9);

for n = 1:9
    Mdl = TreeBagger(450,XTrains, YTrain(:,n),...
                'oobpred', 'On', 'Method', 'regression',...
                'OOBVarImp', 'on');   
    RF_Y_est(:,n) = predict(Mdl, XTrains);
    RF_Y_pred(:,n) = predict(Mdl, XTests);
   if n ~= 3 && n ~= 8
       RF_Y_est(:,n) = round(RF_Y_est(:, n));
        RF_Y_pred(:,n) = round(RF_Y_pred(:, n));
   end
end

%% All Labels
ALL = [RF_Y_est, KNN_Y_est, NN_Y_est, LR_Y_est, SVM_Y_est];
ALL_pred = [RF_Y_pred, KNN_Y_pred,NN_Y_pred, LR_Y_pred, SVM_Y_pred];
ALL_Y_est = zeros(size(XTrain, 1), 9);
ALL_Y_pred = zeros(size(XTest, 1), 9);

for n=1:9
   Mdl = TreeBagger(70, ALL, YTrain(:, n),'Method', 'regression'); 
   ALL_Y_est(:,n) = predict(Mdl,ALL);
   ALL_Y_pred(:,n) = predict(Mdl,ALL_pred);
end

%% Labels
pred_labels = ALL_Y_pred;

end

