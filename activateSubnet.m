function [W_sub, b_sub, idxs] = activateSubnet(W, b, c, Nc, splits, verbose)

L = length(W);

%activate subnetwork based on cluster by slicing out submatrices
W_sub = W;
b_sub = b;
idxs = cell(1, L);
idxs{1} = 1:size(W{2},1);
for r = 2:L
    n = size(W{r},2);               %n - number of units in layer r
    Ns = splits(r);                 %Ns - number of sublayers
    s = ceil(Ns*(c/Nc));            %s - sublayer I want to activate
    idxs{r} = (s-1)*(n/Ns)+1:s*(n/Ns); %neuron indices of sublayer
    
    if splits(r)>1 %note splits(L)==1 always
        W_sub{r} = W{r}(:,idxs{r}); %slice out columns
        W_sub{r+1} = W{r+1}(idxs{r},:); %slice out rows to match columns
        b_sub{r} = b{r}(idxs{r});

        if nargin>5 && verbose
            fprintf('Sublayer %2d/%2d, Cluster %2d/%2d, Units %2d...%2d |', s, Ns, c, Nc, idxs{r}(1), idxs{r}(end))
        end
    end
end
