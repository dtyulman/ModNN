%% load data
%-{
datapath = 'MNIST/';
Ntr = 200; %N samples of each digit
Nte = 150;
Nv = 150;
digits = 0:9; %which digits to load
Ndigits = length(digits);
[Xtr, Ytr, Xv, Yv, Xte, Yte] = loadMNIST(datapath, digits, Ntr, Nv, Nte); %train, valid, test sets
%}
%% evaluate clustering results


for Nc = 0 %[0, 2, 5, 10] %number of clusters. Nc==0 automatically finds optimal number of clusters
    % cluster       
    if Nc == 0
        alpha = 0.0005;
        [Xtr_clust, centroids] = gmeans(Xtr, alpha);
        Nc = size(centroids,1);
    else
        centroid_init = initKmeans(Xtr, Nc, 'initseed.mat');
        [Xtr_clust, centroids] = kmeans(Xtr, Nc, 'Start', centroid_init);
    end   
    cSize = zeros(1,Nc);
    for c = 1:Nc
        cSize(c) = sum(Xtr_clust == c); %number of elements in cluster
    end



%     % plot some samples from each cluster
%     Ns = 5; %plot Ns random samples from each cluster
%     f = figure('Units', 'Inches');
%     f.Position(3) = 7.5;
%     f.Position(4) = 10.5*(Nc/Ndigits);
%     for c = 1:Nc
%         Xtr_c = Xtr(Xtr_clust==c,:);
%         Ytr_c = onehot2class(Ytr(Xtr_clust==c,:));
%         for s = 1:Ns
%             subplot(Nc, Ns, sub2ind([Ns, Nc], s, c))
%             sample=randi(cSize(c));
%             drawMNISTdigit( Xtr_c(sample,:) );
%             if s==1
%                 ylabel(sprintf('Cluster #%d', c))
%             end
%             title(sprintf('Class=%d', Ytr_c(sample)))
%         end
%     end


    % plot centroid of each cluster
    figure;
    for c = 1:Nc
        Nrows = ceil(sqrt(Nc));
        Ncols = round(sqrt(Nc));        
        subplot(Nrows, Ncols, c)
        drawMNISTdigit(centroids(c,:), c, 'Cluster #%d')
    end


    % histogram of classes in each cluster
    Ytrclass = onehot2class(Ytr(Xtr_clust==1,:));
    figure;
    for c = 1:Nc
        Nrows = ceil(sqrt(Nc));
        Ncols = round(sqrt(Nc));
        subplot(Nrows, Ncols, c)
        histogram(Ytrclass(Xtr_clust == c), [0:Ndigits]-0.5);
        xlim([0,Ndigits]-0.5)
        title(sprintf('Cluster #%d (N=%d)', c, cSize(c)))
    end
    
    
    % assign test set to nearest cluster centroid
    Xte_clust = knnsearch(centroids, Xte);

    % histogram of test set placed in each cluster
    Yteclass = onehot2class(Yte);
    figure;
    for c = 1:Nc
        Nrows = ceil(sqrt(Nc));
        Ncols = round(sqrt(Nc));
        subplot(Nrows, Ncols, c)
        histogram(Yteclass(Xte_clust == c), [0:Ndigits]-0.5);
        xlim([0,Ndigits]-0.5)
        title(sprintf('Nearest cluster #%d', c))
    end
    
end





