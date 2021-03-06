% Run this script and it will provide a plot of each mode for the PCA
% evaluation
%
% For each mode user will need to utilize graph to determine how many
% vectors the use which will determine the dimensionality of the eigenspace
% the data will be projected onto.

[frr1, far1, steps] = PCA_Mode1();
[frr2, far2, steps] = PCA_Mode2();

plot(steps,frr1,steps,far1,steps,frr2,steps,far2)