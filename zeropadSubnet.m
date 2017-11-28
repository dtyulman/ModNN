function [W, b] = zeropadSubnet(W_sub, b_sub, W_zero, b_zero, idxs)
%idxs returned from activateSubnet.m

W = W_zero;
b = b_zero;
for r = 2:length(W_sub)   
    W{r}(idxs{r-1},idxs{r}) = W_sub{r}; 
    b{r}(idxs{r}) = b_sub{r};
end