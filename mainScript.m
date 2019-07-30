tic
data = load('training_data.mat');
pred_labels=predict_labels(data.train_inputs(1:815,:),data.train_labels(1:815,:),data.train_inputs(816:1019,:));
error_metric(pred_labels, data.train_labels(816:1019,:))
toc