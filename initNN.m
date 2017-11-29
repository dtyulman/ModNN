function [W, b] = initNN(layers, seedfile)
% layers is length-L array where L is number of layers. Each entry is the number
% of neurons in each layer. First layer is input layer and must be same as dim
% of X datapoints. Final layer is Softmax output layer and must be same as the
% number of classes (i.e. dim of Y datapoints).

if length(layers) < 2
    error('Cannot have less than 2 layers')
end
if any(layers < 1)
    error('Must have at least one neuron per layer')
end

if nargin == 1
    L = length(layers);
    W = cell(1, L);
    b = cell(1, L);
    for r = 2:L
        W{r} = random('Normal', 0, 1/sqrt(layers(r-1)), layers(r-1), layers(r));
    %     b{r} = random('Normal', 0, 1/sqrt(layers(r-1)), layers(r), 1);
        b{r} = zeros(layers(r), 1);
    end
elseif nargin == 2 %TODO: I can actually do this by just seeding the RNG...
    layerStr = sprintf('_%d', layers);
    Wvar = ['W', layerStr];
    bvar = ['b', layerStr];
    S = load(seedfile, Wvar, bvar);
    if length(struct2array(S)) == 2*length(layers) 
        %if dims match up, load them
        eval(sprintf('W=S.%s; b=S.%s;', Wvar, bvar))
        fprintf('Loaded %s, %s\n', Wvar, bvar)
    else
        %if not, they don't exist, so create and save them
        warning('NN with desired size doesn''t exist in seedfile. Creating and appending.')
        if ~isempty(struct2array(S)) %sanity check
            error('initNN - something went wrong loading seed')
        end
        [W, b] = initNN(layers);
        eval(sprintf('%s=W; %s=b;', Wvar, bvar));
        save(seedfile, '-append', Wvar, bvar);
    end
end