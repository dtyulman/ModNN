function ysoft = predictClustNN(x,centroids, W,b,splits)

L = length(W);
Nc = size(centroids,1);

%find nearest centroid to x
distances = zeros(1,Nc);
for i = 1:Nc
    distances(i) = norm(centroids(i,:)-x);
end
[~,c] = min(distances);

% c = knnsearch(centroids, x);


%activate subnetwork based on cluster
[W_sub, b_sub] = activateSubnet(W, b, c, Nc, splits);


a = feedforward(W_sub, b_sub, x);
ysoft = a{L}';