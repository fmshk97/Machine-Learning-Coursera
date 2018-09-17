function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% newTheta = Theta with first value as 0 (i.e. theta(1) = 0)
newTheta = theta;
newTheta(1) = 0;

% Regularization Penalty (Cost)
costPenalty = (lambda / (2 * m)) * (newTheta' * newTheta);

% Cost for logistic regression with regularization
J = ((((-1 .* y)' * log(sigmoid(X * theta))) - ((1 - y)' * log(1 - sigmoid(X * theta)))) / m) + costPenalty;

% Regularization Penalty (Gradient)
gradPenalty = (lambda / m) .* newTheta;

% Gradient for logistic regression with regularization
grad = ((X' * (sigmoid(X * theta) - y)) / m) + gradPenalty;

end
