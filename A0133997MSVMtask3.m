%SVM task 3%
clc
clear all
load('train.mat');
load('test.mat');
load('eval.mat');
eval_predicted=svm(train_data,train_label,test_data,test_label,eval_data,2.1,2,1e-9);

function eval_predicted=svm(train_data,train_label,test_data,test_label,eval_data,c,p,th)
mu = mean(train_data, 2);
stdev = std(train_data, 0, 2);
train_data = bsxfun(@rdivide, bsxfun(@minus, train_data, mu), stdev);
test_data = bsxfun(@rdivide, bsxfun(@minus, test_data, mu), stdev);
eval_data=bsxfun(@rdivide, bsxfun(@minus, eval_data, mu), stdev);



% dual
x = train_data;
d = train_label;
f = -ones(1,length(d));
lb = zeros(length(d),1);
ub = c*ones(length(d),1);
aeq = d';
beq = 0;


H = (d*d').*((x'*x+1).^p);


options = optimset; 
options.LargeScale = 'off';
options.MaxIter = 1000;
alpha = quadprog(H,f,[],[],aeq,beq,lb,ub,[],options);

idx = find(alpha<=ub&alpha>th);

b = mean(train_label(idx)-d'.*(x(:,idx)'*x+1).^p*alpha);

% K=((x'*x+1).^p);
% bs = 1 ./train_label(idx)' - sum(bsxfun(@times, alpha .* d, K(:,idx)),1);
% b = mean(bs);

% Mdl = fitcsvm(train_data',train_label,'KernelFunction','linear');
% [label,score] = predict(Mdl,test_data');

label = sign((alpha(idx).*d(idx))'*((train_data(:,idx)'*train_data+1).^p)+b)';
train_accu = sum(train_label==label)/length(train_label)

label = sign((alpha(idx).*train_label(idx))'*((train_data(:,idx)'*test_data+1).^p)+b)';
test_accu = sum(test_label==label)/length(test_label)

eval_predicted = sign((alpha(idx).*train_label(idx))'*((train_data(:,idx)'*eval_data+1).^p)+b)';


end
