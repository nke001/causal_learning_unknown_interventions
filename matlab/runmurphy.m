clear all

nodeArity   = csvread('dump/nodeArities.csv');
clampedMask = transpose(csvread('dump/clampedMask.csv'));
data        = transpose(csvread('dump/samples.csv'));
dag         = transpose(csvread('dump/dag.csv'));


% data is size d*N (nodes * cases), values are {1,2}
nNodes   = size(data,1);
N        = size(data,2);
maxFanIn = nNodes - 1;


% fan-in
%     L1  L2
% L1   0   1    % only 1 intervention parent allowed for L2 nodes
% L2   0   max  % only max # paretns allowed in total for L2 nodes
maxFanIn_uncertain  = [0 1; 0 maxFanIn];
layering            = [2*ones(1,nNodes) ones(1,nNodes)];
nodeArity_uncertain = [nodeArity nodeArity];
data_uncertain      = [data; clampedMask(nNodes+1:nNodes*2,:)+1];


%
% Compute Edge Probabilities
%
aflp_uncertain  = mkAllFamilyLogPrior  (nNodes*2, ...
                                        'maxFanIn',     maxFanIn_uncertain, ...
                                        'nodeLayering', layering);

aflml_uncertain = mkAllFamilyLogMargLik(data_uncertain, ...
                                        'nodeArity',            nodeArity_uncertain, ...
                                        'clampedMask',          [zeros(nNodes,N); ones(nNodes,N)], ...
                                        'impossibleFamilyMask', aflp_uncertain~=-Inf, ...
                                        'verbose', 0);

epDP_uncertain  = computeAllEdgeProb   (aflp_uncertain, aflml_uncertain);


%
% Plotting
%
figure(1); clf
subplot(1,2,1)
imagesc(dag,            [0 1]);
title('ground truth')
colorbar

subplot(1,2,2)
imagesc(epDP_uncertain, [0 1]);
title('edge marginals (DP) - uncertain')
colorbar;

