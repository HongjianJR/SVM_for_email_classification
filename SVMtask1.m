%Neural Network Part2 Project1 SVM A0133997M CUI HONGJIAN%
clc
clear all
load('train.mat');
load('test.mat');
%train_data 57x2000 double
%train_label 2000x1 double
%test_data 57x1536 double
%test_label 1536x1 double

%data processing%
mutrain=mean(train_data,2);
sigmatrain= std(train_data, 0, 2);
train_Data = bsxfun(@rdivide, bsxfun(@minus, train_data, mutrain), sigmatrain);
test_Data = bsxfun(@rdivide, bsxfun(@minus, test_data, mutrain), sigmatrain);

%i hard margain SVM with linear kernel%
x=train_Data;
d=train_label;
f=-ones(1,length(d));
lb=zeros(length(d),1);
ub=1e4*ones(length(d),1);
aeq=d';
beq=0;
H=(d*d').*(x'*x);
options=optimset;
options.LargeScale='off';
options.MaIter=1000;
alpha=quadprog(H,f,[],[],aeq,beq,lb,ub,[],options);
alpha=roundn(alpha,-9);
w=zeros(size(x,1),1);
for i=1:length(d)
    w=w+alpha(i)*d(i)*x(:,i);
end
idx=find(alpha<=ub&alpha>0);
b=mean(1./train_label(idx)'-w'*train_Data(:,idx));
%g=wx+b%
label = sign(w'*train_Data+b)';
train_accu = sum(train_label==label)/length(train_label)

label = sign(w'*test_Data+b)';

% Mdl = fitcsvm(train_data',train_label,'KernelFunction','linear');
% [label,score] = predict(Mdl,test_data');
test_accu = sum(test_label==label)/length(test_label)



