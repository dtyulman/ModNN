function yhat = classifyClustNN(x,centroids, W,b,splits)
%returns one-hot vector for the class

if isvector(x)
    ysoft = predictClustNN(x,centroids, W,b,splits);

    yhat = zeros(size(ysoft));
    [~,m] = max(ysoft);
    yhat(m) = 1;
else %...there's probably a smarter (vectorized) way to do this 
    yhat = nan(size(x,1), length(b{end}));
    for i = 1:size(yhat,1)
        yhat(i,:) = classifyClustNN(x(i,:),centroids, W,b,splits);
    end
end