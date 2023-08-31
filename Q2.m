%% EE798L: Machine Learning for Wireless Communications
% MATLAB Assignment-1: Linear modelling - least squares and maximum likelihood approach
% NAME: S.Srikanth Reddy; Roll No: 22104092
%Question 2

clear all;
clc;

%generating large samples of x below to fit the polynomial as a continous curve
xlarge=(0:0.0001:1)';
largeN=length(xlarge);

%generating data set below
N=6;
x=unifrnd(0,1,[N,1]);
x=sort(x);

t=zeros(N,1);
for i=1:N
    t(i)=2*x(i)-3+normrnd(0,sqrt(3));
end

%fifth order model below
X5=[ones(N,1) x x.^2 x.^3 x.^4 x.^5];
lambda=[0 10^-6 0.01 0.1]; %given lambda values

X5large=[ones(largeN,1) xlarge xlarge.^2 xlarge.^3 xlarge.^4 xlarge.^5];
tlarge=zeros(largeN,length(lambda));

w_hat5 = zeros(6,length(lambda));
for j=1:length(lambda) %for each lambda
    w_hat5(:,j)=(inv(X5'*X5+N*lambda(j)*eye(6)))*X5'*t; % regularized least squares approach to find estimate w
    tlarge(:,j)=X5large*w_hat5(:,j); %finding corresponding response t using large samples

    subplot(2,2,j)
    plot(x,t,'kx')
    hold on;
    grid on;
    plot(xlarge,tlarge(:,j),'b--')
    title('Fitting 5th order polynomial using Regularized least squares')
    xlabel('x')
    ylabel('t')
    legend('data set',['fit with λ = ',num2str(lambda(j))],'Location','best')
end
