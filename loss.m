function l = loss(y, yhat) 

if sum(y) ~= 1
    error('Error calculating loss: y must be one-hot')
end
if any(size(y) ~= size(yhat))
    error('Inputs must be same size')
end

l = -log(yhat(y==1)); %cross-entropy
