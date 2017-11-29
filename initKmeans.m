function centroids = initKmeans(Xtr, Nc, seedfile)

if nargin == 2
    [~, centroids] = kmeans(Xtr, Nc, 'MaxIter', 0);
elseif nargin == 3
    Xtr_loaded = struct2array(load(seedfile, 'Xtr'));
    if isempty(Xtr_loaded)
        save(seedfile, '-append', 'Xtr')
    else
        if size(Xtr) ~= size(Xtr_loaded) || ~all(Xtr(:) == Xtr_loaded(:))
            error('This seed file is for a different training set')
        end
    end
    
    Cvar = sprintf('centroids_%d', Nc);
    centroids = struct2array(load(seedfile, Cvar));
    if isempty(centroids)
        warning('Centroids with desired Nc doesn''t exist in seedfile. Creating and appending.')
        centroids = initKmeans(Xtr, Nc);
        eval(sprintf('%s=centroids;', Cvar));
        save(seedfile, '-append', Cvar)
    else
        fprintf('Loaded %s\n', Cvar)
    end
end


    