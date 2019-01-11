function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X 12x2
% y 12x1
% theta 2x1

J = 1/(2*m) * sum((X*theta - y).^2) + lambda/(2*m) * (theta(2:end))' * theta(2:end);

temp = theta;
grad(1) = 1/m * (X*temp - y)' * X(:,1);
grad(2:end) = 1/m * (X*temp - y)' * X(:,2:end) + lambda/m * temp(2:end)';


% =========================================================================

grad = grad(:);

end
