function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

%mu = 1/m * sum(X) % [14.112, 14.998] also would work
mu = mean(X); % [14.112, 14.998]

% sigma2 = var(X); % [1.8385, 1.7153] %var does 1/(m-1) by defect, not 1/m. So this would be incorrect for the assignment.
% sigma2 = 1/(m-1) * sum((X - mu).^2); % [1.8385, 1.7153] Incorrect for the assignment
% sigma2 = var(X) * (m-1)/m; % [1.8326 1.7097]
sigma2 = 1/m * sum((X - mu).^2); % [1.8326 1.7097]



% =============================================================


end
