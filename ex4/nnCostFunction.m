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

% X 5000x400
% y 5000x1
% Theta1 25x401
% Theta2 10x26
% nn_params 10285x1

aux = 1:num_labels; %1x10
aux2 = ones(m,1); %5000x1
aux3 = aux' * aux2'; %10x5000

y_vect = y == aux;
% y_vect = 5000x10


X = [ones(size(X,1),1) X]; % X 5000x401
z_2 = X*Theta1'; % z_2 5000x25
a_2 = sigmoid(z_2); % a_2 5000x25
a_2 = [ones(size(a_2,1),1) a_2]; % a_2 5000x26
a_3 = sigmoid(a_2 * Theta2'); %a_3 5000x10

for i=1:m,
	J = J + 1/m * ( -y_vect(i,:) * log(a_3(i,:)') - (1-y_vect(i,:)) * log(1 - a_3(i,:)') );
end;

%In the exercise they ask you to do it using a for loop. I believe the following expression should be equivalent.
%J = 1/m * ( -y_vect * log(a_3') - (1-y_vect * log(1 - a_3')) );
%Though, when running it, it doesn't work, it's like it's super slow or maybe just wrong.

Theta1_sum = sum(sum(Theta1(:,2:end) .* Theta1(:,2:end)));
Theta2_sum = sum(sum(Theta2(:,2:end) .* Theta2(:,2:end)));
J = J + lambda/(2*m)*(Theta1_sum + Theta2_sum);

for t=1:m,
	delta_3(t,:) = a_3(t,:) - y_vect(t,:); % 5000x10
	delta_2(t,:) = delta_3(t,:) * Theta2(:,2:end); %5000x25
	delta_2(t,:) = delta_2(t,:) .* sigmoidGradient(z_2(t,:)); %5000x25
	Theta1_grad = Theta1_grad + (delta_2(t,:))' * X(t,:); %25x401
	Theta2_grad = Theta2_grad + (delta_3(t,:))' * a_2(t,:); %10x26
end;

%This works but in this case they ask to put it inside the loop. Is it faster this way?
%Theta1_grad = Theta1_grad + delta_2' * X; % 25x401
%Theta2_grad = Theta2_grad + delta_3' * a_2; % 10x26

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

%Regularization
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
