%% load data
%{
tic
fprintf('Loading data... ')
datapath = 'MNIST/';
Ntr = 200; %N samples of each digit
Nv = 150;
Nte = 150;
digits = 0:9; %which digits to load
[Xtr, Ytr, Xv, Yv, Xte, Yte] = loadMNIST(datapath, digits, Ntr, Nv, Nte); %train, valid, test sets
toc
fprintf('\n')
%}     
%% initialize
for Nh1 = 100 %[20 50 100] %number of hidden units in layer
    % convert raw data to "neural network" space
    %{
    tic
    fprintf('Preprocessing...\n')
    eta_m2n = .005;
    thres_m2n = 1e-6;
    batchSize_m2n = 100;
    maxIters_m2n = 5000;
    
    dimRep = 100; %dimension of NN representation
    layers_m2n = [size(Xtr,2), dimRep, Nh1, size(Ytr,2)];
    
    layerRep = 2; %layer of NN to use for representation
    if layers_m2n(layerRep) ~= dimRep %sanity check
        error('Dimension of representation not equal to the dimension of the layer number')
    end
    [Xtr, Xv, Xte, W_m2n, b_m2n] = mnist2nn(layers_m2n, Xtr, Ytr, Xv, Yv, Xte, layerRep, eta_m2n, thres_m2n, batchSize_m2n, maxIters_m2n);
    toc
    fprintf('\n')
    %}
    for Nc = [1,10] %[1, 2, 5, 10] %number of clusters
        if mod(Nh1/Nc,1) ~= 0
            warning('Skipping Nh1=%d, Nc=%d. Must be evenly divisible.', Nh1, Nc)
            continue 
        end
        
        eta = .005;
        thresConv = -inf; %1e-8;
        if thresConv == -inf
            Xv = Xv(1,:); %ignore validation set because running to maxIters regardless
            Yv = Yv(1,:); %need to leave one entry so that it doesn't throw error TODO: fix this
        end     
        batchSize = 100;
        maxIters = 1000;
        
        layers = [size(Xtr,2), Nh1, size(Ytr,2)]; %rth layer has <layers(r)> units
        splits = [1 Nc 1]; %rth layer has <splits(r)> sublayers with <layers(r)>/<splits(r)> units each (must be divisible)
        
        [W_init, b_init] = initNN(layers, 'initseedpreproc.mat');
        centroid_init = initKmeans(Xtr, Nc, 'initseedpreproc.mat');
        
        Wlim = [-0.1, 0.1]; %clamp weights between Wlim
        binW = false; %binarize weights
        f = 0.5; %only allow f largest dldWs to be nonzero
        
        %% Sanity check - run network of size Nh1/Nc on just one of the clusters      
%         % cluster training set and assign test/valid set to nearest cluster centroid       
%         [Xtr_clust, centroids] = kmeans(Xtr, Nc, 'Start', centroid_init);
%         Xv_clust = knnsearch(centroids, Xv);
%         Xte_clust = knnsearch(centroids, Xte);
%         
%         % use data from only one cluster for training/testing
%         singleCluster = 2;    
%         Xtr = Xtr(Xtr_clust==singleCluster, :);
%         Ytr = Ytr(Xtr_clust==singleCluster, :);
%         Xv = Xv(Xv_clust==singleCluster, :);
%         Yv = Yv(Xv_clust==singleCluster, :);
%         Xte = Xte(Xte_clust==singleCluster, :);
%         Yte = Yte(Xte_clust==singleCluster, :);

        %% train
        tic
        fprintf('Training...\n')
        [W, b, centroids, riskValid, niters, thres_f, fthres, fraw] = ...
            trainClustNN(W_init, b_init, Xtr, Ytr, Xv, Yv, centroid_init, splits, eta, Wlim, binW, f, thresConv, batchSize, maxIters);
        toc
        fprintf('\n')
        
        %% test            
%         tic
%         errorTrain = nan(1,length(W));
%         errorTest = nan(1,length(W));
%         fprintf('Computing errors...\n')
%         computeTheseErrs = 1:max(1,round(length(W)/100)):length(W);
%         for i = computeTheseErrs
%             errorTrain(i) = errorNN(Ytr, classifyClustNN(Xtr,centroids, W{i},b{i},splits));
%             errorTest(i)  = errorNN(Yte, classifyClustNN(Xte,centroids, W{i},b{i},splits));
%             fprintf('Iter: %g, Train: %g, Test: %g\n', i, errorTrain(i), errorTest(i))
%         end
%         toc

        %% Summarize params and results
        layersStr = sprintf('%d-', layers); layersStr = layersStr(1:end-1);
        splitsStr = sprintf('%d-', splits); splitsStr = splitsStr(1:end-1);
        paramsStr = sprintf('Layers: [%s], Splits: [%s]', layersStr, splitsStr);
%         resultStr = sprintf('Iters=%d, E_{train}=%g, E_{test}=%g', niters, errorTrain(end), errorTest(end));
        miscStr   = sprintf('W[%g_%g]_f%g', Wlim(1), Wlim(2), f);
        miscStrT  = sprintf('W\\in[%g,%g]', Wlim(1), Wlim(2));     
        if binW
            miscStrT = ['Binary\DeltaW, ' miscStrT];
            miscStr = ['binDelta_ ' miscStr];
        end
        saveStr = sprintf('layers%s_splits%s_%s', layersStr, splitsStr, miscStr);
%         
%         save(saveStr, '-v7.3')

        % create matrix where column i is all the params at iteration i concatenated
        % and similarly with only the weights W (not offsets b) 
        theta = nan(length(packNN(W{1},b{1})), length(W));
        thetaW = nan(length(packNN(W{1},cell(size(W{i})))), length(W));
        for i = 1:length(W)
            theta(:,i) = packNN(W{i},b{i});
            thetaW(:,i) = packNN(W{i}, cell(size(W{i})));
        end

        % correlation of params and deltas
        tic
        disp('Correlating...')
        R = corrcoef(theta);
        Rdiff = corrcoef(diff(theta')');
        toc

        % magnitude of params and deltas
        tic
        disp('Magnitude...')
        thetaNorm = sqrt( sum(theta.^2,1) )';
        thetaDiffNorm = sqrt(sum(diff(theta).^2,1))';
        thetaDiffZeros = sum(diff(theta)==0);
        toc
        
        %sparsity of deltas
        fthres_eff = sum(diff(thetaW,[],2)~=0)/length(diff(thetaW));
        
        %% plot results
        fig=figure;

        subplot(3,2,1)
        plot(thres_f)
        xlabel('Iteration')
        ylabel('Thres')
        title(sprintf('Thres to make f=%g',f))
%         plot(riskValid)
%         xlabel('Iteration')
%         ylabel('Total loss on validation set')
%         xlim([1,length(W)])

%         subplot(3,2,2)
%         plot(computeTheseErrs, errorTrain(computeTheseErrs))
%         hold on
%         plot(computeTheseErrs, errorTest(computeTheseErrs))
%         xlabel('Iteration')
%         ylabel('Error')
%         legend({'Train', 'Test'})
%         title(sprintf('%s %s\n%s Batch=%d',paramsStr, miscStrT, resultStr, batchSize))
%         xlim([1,length(W)])

        ax(1) = subplot(3,2,3);
%         imagesc(R)
%         title('w(t) correlation')
        plot(fthres_eff); hold on
        plot(fthres)
        plot(fraw)
        plot(f*ones(size(fthres)), 'k--')
        xlim([1,length(W)])
        title('Fraction of nonzero dldW')
        xlabel('Iteration')
        ylabel('f')

        ax(2) = subplot(3,2,5);
        imagesc(Rdiff)
        title('\Deltaw(t) correlation')
%         linkaxes(ax);
        clear ax

        ax(1) = subplot(3,2,4);
        plot(thetaNorm)
        title('||w(t)||')
        ax(2) = subplot(3,2,6);
        plot(thetaDiffNorm/length(theta))
        title('||\Deltaw(t)||')
        xlabel('Iteration number')
        linkaxes(ax,'x');
        clear ax
        xlim([1,length(W)])
        
%         tic
%         fprintf('Saving...')
%         fig.PaperPosition = [0,0,8.5,11];
%         savefig([saveStr '.fig'])
% %        close(fig)
%         toc
    end
end
