function [Xtrnn, Xvnn, Xtenn, W, b, riskValid, niters] = mnist2nn(layers, Xtr, Ytr, Xv, Yv, Xte, r, eta, thres, batchSize, maxIters)
% preprocesses raw training data to project it into "neural network" space
 
%use defaults from trainClustNN.m
if nargin < 11
    maxIters = [];
end
if nargin < 10
    batchSize = [];
end
if nargin < 9
    thres = [];
end
if nargin < 8
    eta = [];
end
if nargin < 7 || isempty(r)
    r = 2; %layer number that contains the representation of Xtr in "neural network" space
end

if size(Xtr,2) == 748
    error('Input Xtr dimension is not the dimension of MNIST')
end

% load or train fully connected network on training data
seedfile = 'mnist2nn.mat';
layerStr = sprintf('_%d', layers);
Wvar = ['W', layerStr];
bvar = ['b', layerStr];
S = load(seedfile, Wvar, bvar);
if length(struct2array(S)) == 2*length(layers) %if dims match up, load them
    eval(sprintf('W=S.%s; b=S.%s;', Wvar, bvar))
    fprintf('mnist2nn: loaded %s, %s\n', Wvar, bvar)
else 
    warning('mnist2nn: no saved network (%s, %s) found. Training.', Wvar, bvar)
    [W_init, b_init] = initNN(layers, 'initseed.mat');
    
    [W, b, ~, riskValid, niters] = ...
                trainClustNN(W_init, b_init, Xtr, Ytr, Xv, Yv, [], [], eta, [], [], [], thres, batchSize, maxIters);
            
    W = W{end};
    b = b{end};
    
    eval(sprintf('%s=W; %s=b;', Wvar, bvar));
    save(seedfile, '-append', Wvar, bvar);
end



% for each data point, read out the activities of the first hidden layer to
% get representation of data point in "neural network" space
Xtrnn = nan(size(Xtr,1), length(b{2})); %[# datapoints] x [# units in 2nd layer] 
for i = 1:size(Xtrnn,1)
    a = feedforward(W, b, Xtr(i,:));    
    Xtrnn(i,:) =  a{r}';
end
Xtenn = nan(size(Xte,1), length(b{2})); %[# datapoints] x [# units in 2nd layer] 
for i = 1:size(Xtenn,1)
    a = feedforward(W, b, Xte(i,:));    
    Xtenn(i,:) =  a{r}';
end
Xvnn = nan(size(Xv,1), length(b{2})); %[# datapoints] x [# units in 2nd layer]
for i = 1:size(Xvnn,1)
    a = feedforward(W, b, Xv(i,:));    
    Xvnn(i,:) =  a{r}';
end

% Normalize to range [-1, 1]
% normMax = max([Xtrnn(:); Xvnn(:); Xtenn(:)]);
% normMin = min([Xtrnn(:); Xvnn(:); Xtenn(:)]);
% Xtrnn = 2*Xtrnn/(normMax-normMin)-1;
% Xvnn = 2*Xvnn/(normMax-normMin)-1;
% Xtenn = 2*Xtenn/(normMax-normMin)-1;



