function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
 %      Theta1                25x401                    80200  
 %       Theta2                10x26                      2080  
 %       X                   5000x400                 160000
 %        T                   5000X400 ==> i made it.
T = [ones(m,1) X];
z2=T*Theta1';%5000X26
a2 = sigmoid(z2);
A2 = [ones(m,1) a2];%
z3=A2*Theta2';
a3 = sigmoid(z3);

[Max, Index] = max(a3, [], 2);
p=Index;





% =========================================================================


end
