function R = risk(X,Y,centroids, W,b,splits)

F = @(x) predictClustNN(x,centroids, W,b,splits); 

R = 0;
for i = 1:size(Y,1)
    R = R + loss( Y(i,:), F(X(i,:)) );
end
R=R/length(Y);