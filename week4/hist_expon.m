% Generate 10,000 random numbers that are uniformly
% distributed in (0,1)
n = 10000;
u = rand(n,1);

% To 10,000 numbers that are exponentially distributed
% using inverse transformation method
% If u is uniformly distributed 
% Then x = -log(1-u)/lambda is exponentially distributed with rate lambda
lambda = 2;
x = -log(1-u)/lambda;

% To check the numbers are really exponentially distributed
% Plot an histogram of the number 
nb = 50; % Number of bins in histogram 
[n_hist,x_hist] = hist(x,nb);

% We now plot the expected distribution
bin_width = x_hist(2)-x_hist(1);
% lower limit of the bins
lower = x_hist - bin_width/2;
upper = x_hist + bin_width/2;
% expected number of exponentially distributed numbers in each bin
y_expected = n*(exp(-lambda*lower)-exp(-lambda*upper));

% plot the histogram and expected distribution
bar(x_hist,n_hist);
hold on
plot(x_hist,y_expected,'r-','Linewidth',3)
hold off
title('Histogram of 10^4 exponentially distributed psuedo-random numbers')

% print -dpng hist_expon