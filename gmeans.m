function [idx, newC] = gmeans(X, alpha, oldC)
% Implementation of G-means clustering. See "Learning the K in K-means" by
% Hamerly and Elkan
%
% Inputs:
%   X: N-by-P data matrix (N rows/datapoints, P-dimensional)
%   alpha: confidence level. Make sure to do Bonferroni correction.
%   C_init: (optional) cluster centers. Defaults to mean(X);
% Outputs:
%   idx: N-by-1 vector with cluster indices of each point
%   newC: K-by-P matrix of cluster centers


dist = 'sqeuclidean';
% dist = 'correlation';


if nargin < 3
    oldC = mean(X);
end
k = size(oldC,1);

kmax = 100; %more than 100 clusters would be ridiculous, probably. Update: guess not...

all_splits_rejected = false;
while ~all_splits_rejected %&& k < kmax;
    [idx, oldC] = kmeans(X, [], 'Start', oldC, 'Distance', dist);
    all_splits_rejected = true;
    newC = [];
    for i = 1:k %check each cluster if it needs to be split in two
        Xsub = X(idx==i,:); %subset of X belonging to the i-th cluster
        c = oldC(i,:); %center of i-th cluster
        if size(Xsub,1) > 1
            [c1, c2] = split_center(Xsub, c); %initialize two potential centers
            [~, Csub] = kmeans(Xsub, 2, 'Start', [c1; c2], 'Distance', dist);        
            Xp = project(Xsub, Csub(1,:), Csub(2,:));
            if ~isnormal(Xp, alpha)
                all_splits_rejected = false;
                newC(end+1,:) = Csub(1,:);
                newC(end+1,:) = Csub(2,:);
                k = k+1;
            else
                newC(end+1,:) = c;
            end 
        else
            newC(end+1,:) = c;
        end
    end 
    oldC = newC;   
    
    fprintf('gmeans: alpha=%2f, k=%d\n', alpha, k)
    
%     if k >= kmax
%         warning('Max number of clusters reached.')
%     end
end


function [c1, c2] = split_center(X, c)
[coeff,~,latent] = pca(X);
m = coeff(:,1)*sqrt(2*latent(1)/pi);
c1 = c+m';
c2 = c-m';


function Xp = project(X, c1, c2)
% c1, c2 are D-dimensional row vectors
% x is N-by-D
% xp is N-by-1

v = c1-c2;
V = repmat(v, size(X,1), 1);
Xp = dot(X', V')./norm(v)^2;


function tf = isnormal(x, alpha)
%using the Anderson-Darling Test

n = length(x);
z = normcdf(x, mean(x), std(x));
z = reshape(z,n,1);
z = sort(z);
w = 2*(1:n) - 1;
ADStat = -w*(log(z)+ log(1-z(end:-1:1)))/n - n;
ADStat = ADStat*(1 + 4/n - 25/n^2); %this line doesnt exist in adtest.m

[alphas, CVs] = computeCriticalValues(n);
if alpha < alphas(1) || alpha > alphas(end) %|| ADStat > CVs(1) || ADStat < CVs(end)
    error('Alpha out of range of lookup table.')
end
pp = pchip(log(alphas), CVs);
CV = ppval(pp,log(alpha));

tf = ADStat < CV;


function [alphas, CVs] = computeCriticalValues(n)
%copied from adtest.m

alphas = [0.0005 0.001 0.0015 0.002 0.005 0.01 0.025 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5...
    0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99];

% An improved version of the Petitt method for the composite normal case.
% The model used is A_n = A_inf (1+b_0/n+b_1/n^2), where different
% estimates b_0 and b_1 are used for each significance level. This allows
% us to model the entire range of the distribution.
CVs = ...
    [ 1.5649  1.4407  1.3699  1.3187  1.1556    1.0339    0.8733    0.7519    0.6308    0.5598    0.5092    0.4694    0.4366    0.4084...
    0.3835    0.3611    0.3405    0.3212    0.3029    0.2852    0.2679    0.2506    0.2330    0.2144...
    0.1935    0.1673    0.1296] +...
    [-0.9362 -0.9029  -0.8906  -0.8865  -0.8375   -0.7835   -0.6746   -0.5835   -0.4775   -0.4094   -0.3679   -0.3327   -0.3099   -0.2969...
    -0.2795   -0.2623   -0.2464   -0.2325   -0.2164   -0.1994   -0.1784   -0.1569   -0.1377   -0.1201...
    -0.0989   -0.0800   -0.0598]./n +...
    [-8.3249  -6.6022 -5.6461  -4.9685  -3.2208   -2.1647   -1.2460   -0.7803   -0.4627   -0.3672   -0.2833   -0.2349   -0.1442   -0.0229...
    0.0377    0.0817    0.1150    0.1583    0.1801    0.1887    0.1695    0.1513    0.1533    0.1724...
    0.2027    0.3158    0.6431]./n^2;