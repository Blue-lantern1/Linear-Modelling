%% EE798L: Machine Learning for Wireless Communications
% MATLAB Assignment-1: Linear modelling - least squares and maximum likelihood approach
% NAME: S.Srikanth Reddy; Roll No: 22104092
%Question 1

clear all;
clc;

%generating data set below
N=200;
x=unifrnd(-5,5,[N,1]);
x=sort(x);
t=zeros(N,1);
for i=1:N
    t(i)=1 - 2*x(i) + 0.5*power(x(i),2) + normrnd(0,1);
end
plot(x,t,'k.')

%linear model below
Xl = [ones(N,1) x]; %ones in first column, linear x in second column
w_hatl = (inv(Xl'*Xl))*Xl'*t; %least squares approach to find estimate w

t_lmodel=w_hatl(1)+w_hatl(2)*x;

grid on;
hold on;
plot(x,t_lmodel,'g')

%quadratic model below
Xq = [ones(N,1) x x.^2]; %ones in first column, linear (x) in second column, quadratic (x^2) in third column
w_hatq = (inv(Xq'*Xq))*Xq'*t; %least squares approach to find estimate w

t_qmodel=w_hatq(1)+w_hatq(2)*x+w_hatq(3)*x.^2;

grid on;
hold on;
plot(x,t_qmodel,'b')
xlabel('x')
ylabel('t')
title('Fitting linear and quadratic model using least squares')
legend({'data set','linear model','quadratic model'})