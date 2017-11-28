function R = weightsCorr(W, b)
% W,b - cell arrays where entry t is W (b, resp.) at time t
% R - correlation between every pairwise time step

theta = nan(length(packNN(W{1},b{1})), length(W));
for t = 1:length(W)
    theta(:,t) = packNN(W{t},b{t});
end

R = corrcoef(diff(theta')');

