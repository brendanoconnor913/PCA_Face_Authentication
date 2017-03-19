% Run this script and it will provide a plot of each mode for the PCA
% evaluation

[results1, steps] = PCA_Mode1();
[results2, steps] = PCA_Mode2();

plot(steps,results1,steps,results2)