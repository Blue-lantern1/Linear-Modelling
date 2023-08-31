%% EE798L: Machine Learning for Wireless Communications
% MATLAB Assignment-1: Linear modelling - least squares and maximum likelihood approach
% NAME: S.Srikanth Reddy; Roll No: 22104092
%Question 3

clear all;
clc;

load('OlyMens100m.mat','data');
%data contains two columns, first one being olympic year and second one 
%corresponding winning time
x=data(:,1);
t=data(:,2);
N=length(x);

%linear model below
X1=[ones(N,1) x]; 
K=N/3; %K=27/3 = 9 fold cross validation
w1=zeros(2,K); %for storing estimate in each fold
x_dash=zeros(N-3,K); %for training data(year) in each fold
t_dash=zeros(N-3,K); %for training data(time) in each fold
MSE1 = zeros(1,K); %for mean square error in each fold
e1=zeros(3,K); %for error vector from validation
j=N-3;
for i=1:K
    x_dash(:,i)=[(x(1:j))' (x(j+4:N))']'; %collecting training data
    a=x_dash(:,i); %temporary variable
    t_dash(:,i)=[(t(1:j))' (t(j+4:N))']'; %collecting training data
    b=t_dash(:,i); %temporary variable
    X=[ones(length(a),1) a];
    w1(:,i)=(inv(X'*X))*X'*b; %finding estimate for this training data
    t1=X1*w1(:,i); %calculating all t with this estimate
    e1(:,i)=t1(j+1:j+3)-t(j+1:j+3);  %finding error vector on validation data
    MSE1(i)=meansqr(e1(:,i));  %finding its mean square
    j=j-3;
end
index1 = find(MSE1==min(MSE1)); %locating the index of min MSE
w_hat1=w1(:,index1); %getting the best estimate using the above index
t1_choosen=X1*w_hat1; %calculating t with this estimate.
subplot(1,2,1)
plot(x,t,'k.')
hold on;
plot(x,t1_choosen,'m')
grid on;
xlabel('x')
ylabel('t')
title('K fold cross validation - first order model')
legend('data set','linear model','Location','best')

%fourth order model below
X4=[ones(N,1) x x.^2 x.^3 x.^4];
w4=zeros(5,K); %for storing estimate in each fold
MSE4=zeros(1,K); %for mean square error in each fold
e4=zeros(3,K); %for error vector from validation
lambda=(10^-6)*10.^(0:9); %varying lambda
L=length(lambda);
mMSE_l=zeros(1,L); %for min mean square error for each lambda
w_hat4_l=zeros(5,L);% for best estimate after K fold validation for each lambda
for l=1:L
    j=N-3;
    for i=1:K
        a=x_dash(:,i); %training data collected above in x_dash
        b=t_dash(:,i); %training data collected above in t_dash
        X=[ones(length(a),1) a a.^2 a.^3 a.^4];
        w4(:,i)=(inv(X'*X+(N-3)*lambda(l)*eye(5)))*X'*b; %finding estimate
        t4=X4*w4(:,i); %calculating all t with this estimate
        e4(:,i)=t4(j+1:j+3)-t(j+1:j+3);  %finding error vector on validation data
        MSE4(i)=meansqr(e4(:,i));  %finding its mean square
        j=j-3;
    end
    mMSE_l(l)=min(MSE4); %collecting the min MSE for this lambda value
    index4 = find(MSE4==mMSE_l(l)); %locating the index of min MSE for this lambda
    w_hat4_l(:,l)=w4(:,index4); %getting the best estimate using the above index for this lambda
end
index4_best_l=find(mMSE_l==min(mMSE_l)); %locating the index of min of min(MSE) collected above for all lambda
w_hat4_best_l=w_hat4_l(:,index4_best_l); %getting the best estimate corresponding to this lambda
t4_choosen=X4*w_hat4_best_l; %calculating t with this estimate.
subplot(1,2,2)
plot(x,t,'k.')
hold on;
plot(x,t4_choosen,'m')
grid on;
xlabel('x')
ylabel('t')
title('K fold cross validation - fourth order model')
legend('data set','fourth order','Location','best')