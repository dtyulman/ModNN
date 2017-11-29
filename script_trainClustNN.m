%% load data
%-{
datapath = 'MNIST/';
Ntr = 200; %N samples of each digit
Nv = 150;
Nte = 150;
digits = 0:9; %which digits to load
[Xtr, Ytr, Xv, Yv, Xte, Yte] = loadMNIST(datapath, digits, Ntr, Nv, Nte); %train, valid, test sets
%}

% convert raw data to "neural network" space
%-{
eta_m2n = .005;
thres_m2n = 1e-6;
batchSize_m2n = 100;
maxIters_m2n = 5000;

dimRep = 100;
layerRep = 2;
layers_m2n = [size(Xtr,2), dimRep, Nh1, size(Ytr,2)];
[W_init_m2n, b_init_m2n] = initNN(layers_m2n, 'initseed.mat');

[Xtr, Xv, Xte, W_m2n, b_m2n] = mnist2nn(layers_m2n, Xtr, Ytr, Xv, Yv, Xte, layerRep, eta_m2n, thres_m2n, batchSize_m2n, maxIters_m2n);
%}      
%% initialize
for Nh1 = 100 %[20 50 100] %number of hidden units in layer
    for Nc = 1 %[1, 2, 5, 10] %number of clusters
        if mod(Nh1/Nc,1) ~= 0
            continue
        end
        
        eta = .005;
        thres = -inf; %1e-8;
        if thres == -inf
            Xv = Xv(1,:); %ignore validation set because running to maxIters regardless
            Yv = Yv(1,:); %need to leave one entry so that it doesn't throw error TODO: fix this
        end     
        batchSize = 100;
        maxIters = 10000;
        
        layers = [size(Xtr,2), Nh1, size(Ytr,2)]; %rth layer has <layers(r)> units
        splits = [1 Nc 1]; %rth layer has <splits(r)> sublayers with <layers(r)>/<splits(r)> units each (must be divisible)
        
        [W_init, b_init] = initNN(layers, 'initseedpreproc.mat');
        centroid_init = initKmeans(Xtr, Nc, 'initseedpreproc.mat');
        
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
        [W, b, centroids, riskValid, niters] = ...
            trainClustNN(W_init, b_init, Xtr, Ytr, Xv, Yv, centroid_init, splits, eta, thres, batchSize, maxIters);
        toc       
        %% test
        for splitTest = [0,1] %TODO: remove this, this is not a good idea
            if ~splitTest
                splitsTest=ones(1, length(splits));
            else
                if Nc == 1
                    continue %skip splitting if Nc=1
                end
                splitsTest=splits;
            end
            
            tic
            errorTrain = nan(1,length(W));
            errorTest = nan(1,length(W));
            fprintf('Computing errors...\n')
            computeTheseErrs = 1:max(1,round(length(W)/1000)):length(W);
            for i = computeTheseErrs
                errorTrain(i) = errorNN(Ytr, classifyClustNN(Xtr,centroids, W{i},b{i},splitsTest));
                errorTest(i)  = errorNN(Yte, classifyClustNN(Xte,centroids, W{i},b{i},splitsTest));
                fprintf('Iter: %g, Train: %g, Test: %g\n', i, errorTrain(i), errorTest(i))
            end
            toc
            
            %% plot results
            % Summarize params and results
            layersStr = sprintf('%d-', layers); layersStr = layersStr(1:end-1);
            splitsStr = sprintf('%d-', splits); splitsStr = splitsStr(1:end-1);
            splitsTestStr = sprintf('%d-', splitsTest); splitsTestStr = splitsTestStr(1:end-1);
            paramsStr = sprintf('Layers: [%s], Splits: [%s], TestSplits: [%s]', layersStr, splitsStr, splitsTestStr);
            resultStr = sprintf('Iters=%d, E_{train}=%g, E_{test}=%g', niters, errorTrain(end), errorTest(end));
         
            saveStr = sprintf('layers%s_splits%s_splitTest%s', layersStr, splitsStr, splitsTestStr);
%             save(saveStr)
            
            % create matrix where column i is all the params at iteration i concatenated
            theta = nan(length(packNN(W{1},b{1})), length(W));
            for i = 1:length(W)
                theta(:,i) = packNN(W{i},b{i});
            end
            
            % correlation of params and deltas
            tic
            disp('Correlating...')
            R = corrcoef(theta);
            Rdiff = corrcoef(diff(theta')');
            toc
            
            tic
            disp('Magnitude...')
            % magnitude of params and deltas
            thetaNorm = sqrt( sum(theta.^2,1) )';
            thetaDiffNorm = sqrt(sum(diff(theta).^2,1))'; 
            toc
            %%
            fig=figure;
            
            subplot(3,2,1)
            plot(riskValid)
            xlabel('Iteration')
            ylabel('Total loss on validation set')
            xlim([1,length(W)])
            
            subplot(3,2,2)
            plot(computeTheseErrs, errorTrain(computeTheseErrs))
            hold on
            plot(computeTheseErrs, errorTest(computeTheseErrs))
            xlabel('Iteration')
            ylabel('Error')
            legend({'Train', 'Test'})
            title(sprintf('%s\n%s Batch=%d',paramsStr, resultStr, batchSize))
            xlim([1,length(W)])
            
            ax(1) = subplot(3,2,3);
            imagesc(R)
            title('w(t) correlation')
            ax(2) = subplot(3,2,5);
            imagesc(Rdiff)
            title('\Deltaw(t) correlation')
            linkaxes(ax);
            clear ax
            
            ax(1) = subplot(3,2,4);
            plot(thetaNorm)
            title('||w(t)||')
            ax(2) = subplot(3,2,6);
            plot(thetaDiffNorm)
            title('||\Deltaw(t)||')
            xlabel('Iteration number')
            linkaxes(ax,'x');
            clear ax
            xlim([1,length(W)])
             
% %             fig.PaperPosition = [0,0,8.5,11];
% %             savefig(saveStr)
% %             close(fig)
        end
    end
end
