function [Xnn, W, b, riskValid, niters] = mnist2nn(W_init, b_init, Xtr, Ytr, Xv, Yv, r, eta, thres, batchSize, maxIters)
% preprocesses raw training data to project it into "neural network" space
 
%use defaults from trainClustNN.m
if nargin < 10
    maxIters = [];
end
if nargin < 9
    batchSize = [];
end
if nargin < 8
    thres = [];
end
if nargin < 7
    eta = [];
end
if nargin < 6 || isempty(r)
    r = 2; %layer number that contains the representation of Xtr in "neural network" space
end

% train fully connected network on training data
[W, b, ~, riskValid, niters] = ...
            trainClustNN(W_init, b_init, Xtr, Ytr, Xv, Yv, [], [], eta, thres, batchSize, maxIters);

% for each data point, read out the activities of the first hidden layer to
% get representation of data point in "neural network" space
Xnn = nan(size(Xtr,1), length(b{2})); %[# datapoints] x [# units in 2nd layer] 
for i = 1:size(Xnn,1)
    a = feedforward(W, b, Xtr(i,:));    
    Xnn(i,:) =  a{r}';
end
