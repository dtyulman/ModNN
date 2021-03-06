function [W_all, b_all, centroids, R_all, iter, thres_f, fth, fraw] = ...
    trainClustNN(W_init, b_init, Xtr, Ytr, Xv, Yv, clust, splits, eta, Wlims, binDelta, f, convThres, batchSize, maxIters)
% W is a cell array, so that W{r} is the weight matrix for layer r. If
%   layer r has n neurons and layer r-1 has m neurons, W{r} is m-by-n
%   (i.e. for every one of the n neurons in layer r, there is a length-m weight 
%   vector that weights each of the m inputs from layer r-1)
% b is a cell array so that b{r} corresponds to W{r}, is a length-n vector, 
%   i.e. a bias term for each of the n neurons in layer r. 
%
% Note that if W{r-1} is k-by-m, W{r} must be m-by-n.
%
% W_trained and b_trained is the same dimensions as W and b.
%
% All hidden units are ReLU, all output units are Softmax, loss function is cross-entropy 


%%
% This is getting out of control and needs to be rewritten as OOP or need to do
% some better input parsing with varargin. Existing mods from standard FF NN:
% - Modular (precluster, activate subnetworks)
% - Batch GD instead of SGD
% - Upper/lower bound on W
% - Binarized DeltaW
% - Store all intermediate W,b
% - Sparsify i.e. fraction f of DeltaW is nonzero

%% defaults and convenience vars
L = length(W_init); %number of layers

if nargin < 15 || isempty(maxIters)
    maxIters = 10000;
end
if nargin < 14 || isempty(batchSize)
    batchSize = 1; %default to SGD
end
if nargin < 13  || isempty(convThres)
    convThres = 0.1;
end
if nargin < 12 || isempty(f)
    f = 1; %all Ws nonzero
end
if nargin < 11 || isempty(binDelta)
    binDelta = false;
end
if nargin < 10 || isempty(Wlims)
    Wlims = [-inf, inf];
end
if nargin < 9 || isempty(eta)
    eta = 1;
end
if nargin < 8 || isempty(splits)
    splits = ones(1, L); %no splits
end
if nargin < 7 || isempty(clust)
    clust = 1;
end

if isscalar(clust) %"clust" argument specifies...
    Nc=clust; %...number of clusters (scalar) or...
else
    Nc=size(clust,1); %...or centroid initialization (Nc-by-d) for kmeans.
end

Wmin = Wlims(1);
Wmax = Wlims(2);
if Wmin > -inf || Wmax < inf
    fprintf('Limiting weights to range [%g, %g]\n', Wmin, Wmax)
end
if binDelta
    fprintf('Binarizing dldW\n')
end
if f < 1
    fprintf('Sparsifying to f=%g\n', f)
end

cellfun_add = @(C1,C2){C1+C2};
cellfun_sub = @(C1,C2){C1-C2};
cellfun_scale = @(C){eta/batchSize*C};
cellfun_zero = @(C){0*C};
cellfun_binarize = @(C){sign(C)};
cellfun_boundW = @(C){max(Wmin, min(Wmax, C))};

%% input/sanity checking
if ~isempty(W_init{1})
    error('W indexing starts at 2. Leave first cell empty')
end
if ~isempty(b_init{1})
    error('b indexing starts at 2. Leave first cell empty')
end
for r=2:L-1
    if size(W_init{r},2) ~= size(W_init{r+1},1)
        error(['W{r} and W{r+1} dim mismatch, r=' num2str(r)])
    end
end
for r=2:L
    if size(W_init{r},2) ~= length(b_init{r})
        error(['W{r} and b{r} dim mismatch, r=' num2str(r)])
    end
end 
if size(Xtr,1) ~= size(Ytr,1)
    error('Xtr and Ytr must have same number of datapoints')
end
if size(Xv,1) ~= size(Yv,1)
    error('Xv and Yv must have same number of datapoints')
end

if size(Ytr,2) ~= size(Yv,2)
    error('Y datapoints must be same dim for both datasets')
end
if size(Xtr,2) ~= size(Xv,2)
    error('X datapoints must be same dim for both datasets')
end
if size(Ytr,2) ~= size(W_init{end},2)
    error('W{L} and Y dim mismatch')
end
if size(Xtr,2) ~= size(W_init{2},1)
    error('W{L} and Y dim mismatch')
end
if length(splits) ~= length(W_init)
    error('length(splits) ~= length(W)')
end
if splits(1) ~= 1 || splits(end) ~= 1
    error('Can''t split first or last layer')
end
for r = 2:L %restrict to even splits (TODO: temporary)
    [~,n] = size(W_init{r});
    if rem(n/splits(r),1)~=0
        error(['Must have number of neurons in layer ' num2str(r) ' evenly divisible by number of sublayers'])
    end
end
if any(splits > Nc)
    error('Cannot have more sublayers than clusters')
end
if batchSize > size(Xtr, 1)
    warning('Batch size greater than training set. Setting batch size to length of training set.')
    batchSize = size(Xtr, 1);
end
if Wmin >= Wmax
    error('Lower bound on W must be less than upper bound')
end

%% cluster
if isscalar(clust)
    [Xtr_clust, centroids] = kmeans(Xtr, Nc);
else
    [Xtr_clust, centroids] = kmeans(Xtr, Nc, 'Start', clust);
end

%% train via SGD
W_new = W_init; 
W_all = cell(1,maxIters+1); %store all variables over time
W_all{1} = W_new;

b_new = b_init;
b_all = cell(1,maxIters+1);
b_all{1} = b_new;

R = @(W,b) risk(Xv,Yv,centroids, W,b,splits); %define the error function for early stop
R_new = R(W_new, b_new);
R_all = nan(1,maxIters+1);
R_all(1) = R_new;

thres_f = nan(1,maxIters+1);
fth = nan(1,maxIters+1);
fraw = nan(1,maxIters+1);


% rng(1) %make SGD replicable by seeding the random number generator

converged = false;
for iter = 1:maxIters
    W_old = W_new; 
    b_old = b_new;
    
    %% minibatch gradient
    dldW = cellfun(cellfun_zero, W_new); %initialize gradient to zero
    dldb = cellfun(cellfun_zero, b_new);
    for b = 1:batchSize
        
        if batchSize == size(Xtr, 1)
            i=b;
        else
            i = round(1+(size(Ytr,1)-1)*rand); %select random point (w/ repacement... TODO: no replacement)
        end
        [x,y,c] = deal(Xtr(i,:), Ytr(i,:), Xtr_clust(i));

        %activate subnetwork based on cluster
        verbose = false;
        [W_sub, b_sub, idxs] = activateSubnet(W_old, b_old, c, Nc, splits, verbose);

        %compute gradient with feedforward+backprop
        [dldW_i, dldb_i] = gradient(W_sub, b_sub, x, y);  

        %pad with zeros to restore original shape
        [dldW_i, dldb_i] = zeropadSubnet(dldW_i, dldb_i, cellfun(cellfun_zero, W_new), cellfun(cellfun_zero, b_new), idxs);

        %accumulate the gradient
        dldW = cellfun(cellfun_add, dldW, dldW_i);
        dldb = cellfun(cellfun_add, dldb, dldb_i);
        
        %DEBUG
%         (sum(sum(dldW_i{2})==0)+sum(sum(dldW_i{3})==0))/110 %number of empty columns 
%         dldWpacked = packNN(dldW_i,cell(size(dldW_i)));
%         sum(dldWpacked==0)/length(dldWpacked) %number of zero entries in dldW_i
    end 
       
    %% update parameters
    
    %DEBUG
%     (sum(sum(dldW{2})==0)+sum(sum(dldW{3})==0))/110 %number of empty columns   
%     dldWpacked = packNN(dldW,cell(size(dldW)));
%     sum(dldWpacked==0)/length(dldWpacked) %number of zero entries in batch

    dldWpacked = packNN(dldW,cell(size(dldW)));
    fraw(iter+1) = sum(dldWpacked~=0)/length(dldWpacked);
    if f < 1
        [dldW, thres_f(iter+1), fth(iter+1)] = sparsify(dldW, f);
    end
    if binDelta
        dldW = cellfun(cellfun_binarize, dldW);
    end
    W_step = cellfun(cellfun_scale, dldW);
    W_new = cellfun(cellfun_sub, W_old, W_step);
    W_new = cellfun(cellfun_boundW, W_new);
    
    b_step = cellfun(cellfun_scale, dldb);
    b_new = cellfun(cellfun_sub, b_old, b_step);
    
    W_all{iter+1} = W_new;
    b_all{iter+1} = b_new;
    
    %check convergence
    R_old = R_new;
    R_new = R(W_new, b_new);
    R_all(iter+1) = R_new;  
    disp(['Iter: ' num2str(iter) '  Grad: ' num2str(norm(packNN(dldW, dldb)), '%05f') '  Risk: ' num2str(R_new, '%05f')])
    if abs(R_new - R_old) < convThres
        converged = true;
        break;
    end
end

W_all = W_all(1:iter+1); %crop unused memory slots
b_all = b_all(1:iter+1);
R_all = R_all(1:iter+1);
thres_f = thres_f(1:iter+1);
fth = fth(2:iter+1);
fraw = fraw(2:iter+1);

if ~converged
    warning('Reached maxIters without convergence.')
end

end

%% Helper functions
function [dldW, dldb] = gradient(W, b, x, y)
    L = length(W);
    
    [a,z] = feedforward(W, b, x);
    
    %Backpropagation
    d = cell(1,L);
    d{L} = a{L} - y'; %Softmax output and cross-entropy loss.
    for r = L-1 : -1 : 2
        d{r} = diag((z{r}>=0))*W{r+1}*d{r+1}; %Derivative of ReLU is Indicator
    end    

    %Gradient
    dldW = cell(1,L);
    dldb = cell(1,L);
    for r = 2:L
        dldW{r} = a{r-1}*d{r}';
        dldb{r} = d{r};
    end 
end

%%
function [Wsparse, th, fth] = sparsify(W, f)
% Find a threshold th below which to set entries of cell array W to zero
% such that fraction f of thresholded array Wsparse are nonzero

%concatenate W into a single vector for sanity
[Wvec, layers] = packNN(W,cell(size(W)));
N = length(Wvec);
absWvec = abs(Wvec);

lo = min(absWvec);
hi = max(absWvec);
th = mean(absWvec);
fth = sum(absWvec>th)/N;
while fth ~= f %binary search
    if fth < f
        hi = th;       
    else
        lo = th;
    end
    th = (lo+hi)/2;
    
    fth_new = sum(absWvec>th)/N;
    if fth == fth_new
        %fth is as close to f as it can be, but not equal to it due to resolution of fraction
        break
    end    
    fth = fth_new;
end

Wvec=Wvec.*(absWvec>th);

%reshape W back into a cell array
L = length(layers);
Wsparse = cell(1, L);
t=1;
for r = 2:L
    Wsparse{r} = reshape(Wvec(t:t+layers(r-1)*layers(r)-1), layers(r-1), layers(r));
    t=t+layers(r-1)*layers(r);
end

end
















