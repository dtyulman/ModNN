function e = errorNN(Y, Yhat)

if any(sum(Y,2) ~= 1)
    error('Y must be one-hot')
end
if any(sum(Yhat,2) ~= 1)
    error('Yhat must be one-hot')
end
if any(size(Y) ~= size(Yhat))
    error('Inputs must be same size')
end

e = (sum(sum(Y~=Yhat))/size(Y,2))/size(Y,1);