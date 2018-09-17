function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1), X];

z2 = X * Theta1';
a2 = [ones(m, 1), sigmoid(z2)];

z3 = a2 * Theta2';
h = sigmoid(z3);

y_new = zeros(m, num_labels);

for i=1:m
    y_new(i, y(i)) = 1;
end

t1 = ((-1 .* y_new) .* log(h)) - ((1 - y_new) .* log(1 - h));
J = sum(t1(:)) / m;

theta1_new = Theta1(:, 2:end);
theta2_new = Theta2(:, 2:end);

theta_sum = sum([theta1_new(:); theta2_new(:)]' * [theta1_new(:); theta2_new(:)]);

J = J + ((lambda / (2 * m)) * theta_sum);

delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

for t=1:m
    % Forward propagation
    xt = X(t, 1 : end)';
    yt = y_new(t, 1 : end)';
    a1 = xt;
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);   % a3 = h(xt)
    
    % Backward propagation
    e3 = a3 - yt;
    e2 = (Theta2' * e3) .* [1; sigmoidGradient(z2)];
    e2 = e2(2 : end);  % omitting the bias value
    delta1 = delta1 + (e2 * a1');
    delta2 = delta2 + (e3 * a2');
end

penalty_theta1 = (lambda / m) * [zeros(size(Theta1, 1), 1), Theta1(:, 2 : end)];
penalty_theta2 = (lambda / m) * [zeros(size(Theta2, 1), 1), Theta2(:, 2 : end)];

% Regularized gradient for cost function J
Theta1_grad = (delta1 / m) + penalty_theta1;
Theta2_grad = (delta2 / m) + penalty_theta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
