function [theta, layers] = packNN(W, b)

layers = zeros(1, length(W));
layers(1) = size(W{2},1);
nParams = 0;
for r = 2:length(W)
    layers(r) = size(W{r},2);
    nParams = nParams + numel(W{r}) + numel(b{r});
end

theta = nan(nParams,1);
t = 1;
for r = 2:length(W)
    theta(t:t+numel(W{r})-1) = W{r}(:);
    t = t+numel(W{r});
    
    theta(t:t+numel(b{r})-1) = b{r}(:);
    t = t+numel(b{r});
end
    
    