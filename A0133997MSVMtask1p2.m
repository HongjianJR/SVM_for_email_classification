%SVM task1 ii
%Neural Network Part2 Project1%
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

%i polynomial kernel%
x=train_Data;
d=train_label;
f=-ones(1,length(d));
lb=zeros(length(d),1);
ub=1e4*ones(length(d),1);
aeq=d';
beq=0;  
p=5;
H=(d*d').*((x'*x+1).^p);
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
b = mean(train_label(idx)-d'.*(x(:,idx)'*x+1).^p*alpha);
g = sign((alpha(idx).*d(idx))'*((train_Data(:,idx)'*train_Data+1).^p)+b)';
%g=wx+b%
label=g;
train_accu = sum(train_label==label)/length(train_label)


label=sign((alpha(idx).*train_label(idx))'*((train_Data(:,idx)'*test_Data+1).^p)+b)';
test_accu = sum(test_label==label)/length(test_label)
