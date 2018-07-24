function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];
% 예측 함수를 만들어보자. 이 장에서는 주어진 값 y만을 이용해서, 서로 값을 비교한다.
% 그래서 sigmoid 를 이용해서 사용하는데 가장 최고 높은값을 이용해 해당값을 찾는다.
% 알다시피 sigmoid는 0~1의 값만 가질수 있기때문에, 해당값에서 인덱스 값을 구해서 알수있다.


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
% 
%all_theta=10x401
%X=5000X401

h = sigmoid(X*all_theta')

[Max, Index] = max(h, [], 2)
p = Index;






% =========================================================================


end
