function [a,z] = feedforward(W, b, x)

L = length(W);
z = cell(1,L);
a = cell(size(z));
a{1} = x';
for r = 2:L
    z{r} = W{r}'*a{r-1} + b{r};
    a{r} = max(0, z{r}); %All hidden units are ReLU
end
a{L} = softmax(z{L}); %Output units are Softmax
