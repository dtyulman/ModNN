function [W, b] = unpackNN(theta, layers)

L = length(layers);
W = cell(1, L);
b = cell(1, L);
t=1;
for r = 2:L
    W{r} = reshape(theta(t:t+layers(r-1)*layers(r)-1), layers(r-1), layers(r));
    t=t+layers(r-1)*layers(r);
    
    b{r} = theta(t:t+layers(r)-1);
    t=t+layers(r);
end


