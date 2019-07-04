% COMP9334 
% Week 3B, Database server example

%% 
% After working out the balance equation, we need to solve
% the set of linear equations
% 
%
% Given that the balance equations are linearly dependent,
% there are two methods: 
% 
% There are two methods:
% (1) You can choose any 5 of the 6 equations 
% (2) You can use all 6 equations and use linear least squares

%% Method 1
%
% You can choose any five of the 6 equations on page 18.
% 
% For this illustration, I have chosen to use the first 
% 5 equations on page 28 together
% with sum( probabilities ) = 1 
% 
% In principle, you can choose any of the 5 equations together
% with sum( probabilities ) = 1 
%
% We put the linear equations in standard form A x = b
% where x is the unknown vector  
% 
A = [ 6  -4  -2   0   0   0
     -3  10   0  -4  -2   0
     -3   0   8   0  -4  -2
      0  -3   0   4   0   0
      0  -3  -3   0   6   0
      1   1   1   1   1   1];
b = [0 0 0 0 0 1]';
x = A\b

%% Method 2 
%
% You can use all the six equations and together
% with sum( probabilities ) = 1 
%
% In the following, the first six rows of the 
% matrix A2 contain the coefficients of the six
% equations. The last row is for sum( probabilities ) = 1 
%
A2 = [ 6  -4  -2   0   0   0
      -3  10   0  -4  -2   0
      -3   0   8   0  -4  -2
       0  -3   0   4   0   0
       0  -3  -3   0   6   0
       0   0  -3   0   0   2
       1   1   1   1   1   1];
b2 = [0 0 0 0 0 0 1]';
x2 = A2 \ b2;     



