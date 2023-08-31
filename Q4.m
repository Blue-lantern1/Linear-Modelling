%% EE798L: Machine Learning for Wireless Communications
% MATLAB Assignment-1: Linear modelling - least squares and maximum likelihood approach
% NAME: S.Srikanth Reddy; Roll No: 22104092
%Question 4

clear all;
clc;

u1=[2 1]; %mean vector1
cov_mtx1 = eye(2); %covariance matrix1

u2=u1; %mean vector2
cov_mtx2 = [1 0.8;0.8 1]; %covariance matrix2

x1=[-1.5:0.1:4.5]; %first parameter
y1=[-1.5:0.1:4.5]; %second paramter
%2-D combinations below
[X1,Y1]=meshgrid(x1,y1);
W = [X1(:) Y1(:)];

%finding Multivariate Gaussian
prob1 = mvnpdf(W,u1,cov_mtx1); 
prob1 = reshape(prob1,length(y1),length(x1));

subplot(2,2,1)
surf(x1,y1,prob1)
xlabel('x1')
ylabel('y1')
zlabel('pdf')
title('covariance matrix = [1 0;0 1]')
subplot(2,2,2)
contour(x1,y1,prob1)
xlabel('x1')
ylabel('y1')
title('Contour Plot')

%finding Multivariate Gaussian
prob2 = mvnpdf(W,u2,cov_mtx2);
prob2 = reshape(prob2,length(y1),length(x1));

subplot(2,2,3)
surf(x1,y1,prob2)
xlabel('x1')
ylabel('y1')
zlabel('pdf')
title('covariance matrix = [1 0.8;0.8 1]')
subplot(2,2,4)
contour(x1,y1,prob2)
xlabel('x1')
ylabel('y1')
title('Contour Plot')