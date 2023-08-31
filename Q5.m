%% EE798L: Machine Learning for Wireless Communications
% MATLAB Assignment-1: Linear modelling - least squares and maximum likelihood approach
% NAME: S.Srikanth Reddy; Roll No: 22104092
%Question 5

clear all;
clc;

%generating data set below
N=200;
x=unifrnd(-5,5,[N,1]);
x=sort(x);
t=zeros(N,1);
for i=1:N
    t(i)= 5*x(i)^3 - x(i)^2 + x(i) + normrnd(0,sqrt(300));
end

%generating large samples of x below to fit the polynomial as a continous curve
%and also to calculate mean
xlarge=(-5:0.2:5)';
largeN=length(xlarge);

%linear model below
Xl = [ones(N,1) x];
Xl_large=[ones(largeN,1) xlarge];
A=inv(Xl'*Xl);
w_hatl = A*Xl'*t;
t_lmodel=w_hatl(1)+w_hatl(2)*x;
mean_l=w_hatl(1)+w_hatl(2)*xlarge; %finding corresponding t using large samples which is nothing but mean
var_l=zeros(largeN,1);
for i=1:largeN
    var_l(i)=Xl_large(i,:)*A*Xl_large(i,:)'; 
end
var_l_new=var_l*(t'*t-t'*Xl*w_hatl)/N; %calculating (σ_new)^2
subplot(1,3,1)
plot(x,t,'k.')
hold on;
plot(x,t_lmodel,'k')
errorbar(xlarge,mean_l,var_l_new,'g')
xlabel('x')
ylabel('t')
legend('data set','linear model','error bars','Location','best')
title('predictive variance')

%cubic model below
X3 = [ones(N,1) x x.^2 x.^3];
X3_large=[ones(largeN,1) xlarge xlarge.^2 xlarge.^3];
B=inv(X3'*X3);
w_hat3 = B*X3'*t;
t_3model=X3*w_hat3;
mean_3=X3_large*w_hat3; %finding corresponding t using large samples which is nothing but mean
var_3=zeros(largeN,1);
for i=1:largeN
    var_3(i)=X3_large(i,:)*B*X3_large(i,:)';
end
var_3_new=var_3*(t'*t-t'*X3*w_hat3)/N; %calculating (σ_new)^2
subplot(1,3,2)
plot(x,t,'k.')
hold on;
plot(x,t_3model,'k')
errorbar(xlarge,mean_3,var_3_new,'m')
xlabel('x')
ylabel('t')
legend('data set','cubic model','error bars','Location','best')
title('predictive variance')

%sixth-order model below
X6 = [ones(N,1) x x.^2 x.^3 x.^4 x.^5 x.^6];
X6_large=[ones(largeN,1) xlarge xlarge.^2 xlarge.^3 xlarge.^4 xlarge.^5 xlarge.^6];
C=inv(X6'*X6);
w_hat6 = C*X6'*t;
t_6model=X6*w_hat6;
mean_6=X6_large*w_hat6; %finding corresponding t using large samples which is nothing but mean
var_6=zeros(largeN,1);
for i=1:largeN
    var_6(i)=X6_large(i,:)*C*X6_large(i,:)';
end
var_6_new=var_6*(t'*t-t'*X6*w_hat6)/N; %calculating (σ_new)^2
subplot(1,3,3)
plot(x,t,'k.')
hold on;
plot(x,t_6model,'k')
errorbar(xlarge,mean_6,var_6_new,'b')
xlabel('x')
ylabel('t')
legend('data set','sixth-order','error bars','Location','best')
title('predictive variance')