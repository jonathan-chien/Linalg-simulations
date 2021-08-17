%% Case 1: Clusters of conditions across neurons

clearvars
rng('shuffle')

% Set paramaters.
nCondClusters    = 2;
nCondsPerCluster = 30;
nNeurons         = 200;
firingRateRange  = [2 18]; % used only with randi method
noiseFactor      = [1 1 1 1 1 1 1 1 1 1]*1; % length must equal or exceed nCondClusters
manualShuffle    = false;
PEV              = 0.8;
nDims            = 3;
plotAs           = 'points';
pcaVecType       = 'loadings';
currentCase      = 1;

% Construct data matrix.
% vectors = constructCondClusters(nCondClusters, nCondsPerCluster, nNeurons,...
%                                 'firingRateRange', firingRateRange,...
%                                 'construction', 'randn',...
%                                 'noiseFactor', noiseFactor);
switch pcaVecType
    case 'loadings'
        vectors = construct_clusters(nCondClusters, nCondsPerCluster, nNeurons,...
                                     'randiRange', firingRateRange,...
                                     'construction', 'randn',...
                                     'noiseFactor', noiseFactor, ...
                                     'clustersIn', 'rows');
    case 'prinComps'
        vectors = construct_clusters(nCondClusters, nCondsPerCluster, nNeurons,...
                                     'randiRange', firingRateRange,...
                                     'construction', 'randn',...
                                     'noiseFactor', noiseFactor, ...
                                     'clustersIn', 'columns');
        vectors = normalize(vectors, 2);
end
        
% Option to shuffle data at this point.                               
if manualShuffle
    for i = 1:size(vectors,2)
        vectors(:,i) = vectors(randperm(size(vectors,1)),i);
    end
end

% Perform PCA.
pcaOut = pca_jmc(vectors, 'corrMatrix', true, 'PEV', PEV, 'nDims', nDims);

% Visualize data.
plotSimData(vectors, pcaOut, plotAs, pcaVecType)


%% Case 2: Clusters of neurons across conditions

clearvars
rng('default')

% Set paramaters.
nNeuronsPerCluster = 50;
nClusters          = 3;
nCond              = 12;
firingRateRange    = [2 18];
noiseFactor        = [2 2 2];
manualShuffle      = false;
PEV                = 1;
plotAs             = 'points';
pcaVecType         = 'loadings';
currentCase        = 2;

% Construct data matrix.
vectors = constructNeuronClusters(nNeuronsPerCluster, nClusters, nCond,...
                                  'firingRateRange', firingRateRange,...
                                  'construction', 'randn',...
                                  'noiseFactor', noiseFactor);

% Option to shuffle data at this point.   
if manualShuffle
    for i = 1:size(vectors,2)
        vectors(:,i) = vectors(randperm(size(vectors,1)),i);
    end
end                              

% Perform PCA.
pcaOut = pca_jmc(vectors, 'corrMatrix', true, 'PEV', PEV);

% Visualize data.
plotSimData(vectors, pcaOut, plotAs, pcaVecType)


%% Case 3: Non-clustered low-rank data (rows x neurons)
% Rank deficiency engineered in rows using SVD.

clearvars 
rng('shuffle')

% Set parameters.
lowRank          = 3; % dimensionality of linear manifold/affine space, not vector subspace (see 'affine' nvp of constructLowRankMatrix1 for more info).
nConds           = 12;
nNeurons         = 150;
noiseFactor      = 1;
weightConstraint = 1;
basisMethod      = 'randn'; % orth, gramschmidt, or randn
removeMethod     = 'randperm'; % 'randperm', 'leading', 'trailing'
manualShuffle    = false;
PEV              = 0.8;
plotAs           = 'points';
pcaVecType       = 'loadings';
currentCase      = 3;

construction = 1;
switch construction
    case 1
        vectors = constructLowRankMatrix1(lowRank, nConds, nNeurons,...
                                          'remove', removeMethod,...
                                          'noiseFactor', noiseFactor);
    case 2
        vectors = constructLowRankMatrix2(lowRank, nConds, nNeurons,...
                                          'basis', basisMethod,...
                                          'noiseFactor', noiseFactor,...
                                          'weightConstraint', 1);
end

% Option to shuffle data at this point.                                 
if manualShuffle
    for i = 1:size(vectors,2)
        vectors(:,i) = vectors(randperm(size(vectors,1)),i);
    end
end   

% Perform PCA.
pcaOut = pca_jmc(vectors, 'corrMatrix', true, 'nDims', 5);

% Visualize data.
plotSimData(vectors, pcaOut, plotAs, pcaVecType)


%% ePAIRS 

fprintf('Current case is %d.\n', currentCase)

test = 'elliptical';
clear params
params.nKNN = 3; 
params.input = 'loadings';
params.prior = 'permutation'; 
params.origVectors = vectors;
params.nBootstraps = 1000;
switch test
    case 'elliptical'
        params.dimSizes = pcaOut.eigenvalues;
        [PAIRSstat, p, dists]...
            = pairsClusterTest_elliptical(pcaOut.loadings, params);
    case 'spherical'
        [PAIRSstat, p, dists]...
            = pairsClusterTest_elliptical(pcaOut.V, params);
end


%% Local functions

function vectors = constructCondClusters(nClusters, nConds, nNeurons, nvp)
                                         
arguments
    nClusters % number of clusters of conditions
    nConds % number of conditions per cluster
    nNeurons
    nvp.firingRateRange = [2 18] % only used if 'construction' = randi
    nvp.construction = 'randn' 
    nvp.noiseFactor = ones(20,1)
end

assert(length(nvp.noiseFactor) >= nClusters,...
       "Length of 'noiseFactor' must be >= nClusters.")

vectors = zeros(nClusters*nConds,nNeurons);
for i = 1:nClusters
    % Construct current cluster of conditions according to specified
    % method.
    switch nvp.construction
        case 'randi'            
            vectors((i-1)*nConds+1 : (i-1)*nConds+nConds, :)...
                = repmat(randi(nvp.firingRateRange,1,nNeurons),nConds,1);
        case 'randn'
            vectors((i-1)*nConds+1 : (i-1)*nConds+nConds, :)...
                = repmat(randn(1,nNeurons)*2+11,nConds,1);
    end
    % Add gaussian noise to the newly created cluster.
    vectors((i-1)*nConds+1 : (i-1)*nConds+nConds, :)...
        = vectors((i-1)*nConds+1 : (i-1)*nConds+nConds, :) +...
          randn(size(vectors((i-1)*nConds+1 :...
                             (i-1)*nConds+nConds, :)))*nvp.noiseFactor(i);
end

end


function vectors = constructNeuronClusters(nNeurons, nClusters, nCond, nvp)

arguments
    nNeurons {mustBeInteger} % number of neurons per cluster
    nClusters {mustBeInteger} % number of clusters
    nCond {mustBeInteger} % number of conditions
    nvp.firingRateRange = [2 20]
    nvp.construction = 'randn' 
    nvp.noiseFactor = ones(50,1)
end

assert(length(nvp.noiseFactor) >= nClusters,...
       "Length of 'noiseFactor' must equal or exceed nClusters.")

vectors = zeros(nCond, nNeurons*nClusters);
for i = 1:nClusters
    % Construct current cluster.
    switch nvp.construction
        case 'randi'
            vectors(:, (i-1)*nNeurons+1 : (i-1)*nNeurons+nNeurons)...
                = repmat(randi(firingRateRange,nCond,1),1,nNeurons);
        case 'randn'
            vectors(:, (i-1)*nNeurons+1 : (i-1)*nNeurons+nNeurons)...
                = repmat(randn(nCond,1)*2+11,1,nNeurons);
    end
    % Add Gaussian noise to newly created cluster.
    vectors(:, (i-1)*nNeurons+1 : (i-1)*nNeurons+nNeurons)...
        = vectors(:, (i-1)*nNeurons+1 : (i-1)*nNeurons+nNeurons) +...
          randn(size(vectors(:, (i-1)*nNeurons+1 :...
                                (i-1)*nNeurons+nNeurons)))*nvp.noiseFactor(i);
end                                                            
                                       
end


function vectors = constructLowRankMatrix1(lowRank, nConds, nNeurons, nvp)
% Uses SVD. Note that thte matrix returned here will actually have rank of
% lowRank+1, even though the intrinsic dimensionality of the manifold is
% equal to lowRank. This helps account for the loss of a degree of freedom
% when mean-centering in the pre-processing phase of PCA.

arguments
    % lowRank = number of vectors in basis for row space prior to addition
    % of noise and translation away from origin.
    lowRank {mustBeInteger} 
    nConds {mustBeInteger}
    nNeurons {mustBeInteger}
    nvp.remove = 'randperm' % 'randperm' or 'trailing'
    nvp.noiseFactor = 1
    nvp.affine (1,2) = [3 11] % logical false or vector of length 2 (scale factor and translation)
end

% Create multivariate Gaussian and perform SVD.
vectors = randn(nConds, nNeurons); 
[U,S,V] = svd(vectors);

% Option to either randomly set nConds-lowRank singular values to 0 or set
% trailing nConds-lowRank singular values to 0.
assert(~isempty(nvp.remove), "Must specify value for 'remove'.")
switch nvp.remove
    case 'randperm'
        remove = randperm(nConds, nConds-lowRank);
    case 'leading'
        remove = 1 : nConds - lowRank;
    case 'trailing'
        remove = lowRank+1 : nConds;
end
S(sub2ind([nConds nNeurons], remove, remove)) = 0;

% Reconstruct matrix.
vectors = (U*S*V');

% Option to translate point cloud away from origin. Note that in so doing
% we add a degree of freedom here and create a proper affine space (we do
% this as this more closely mimics real neural firing rates). This inflates
% by 1 the rank of the data when considered as residing within a proper
% vector subspace. The intrinsic dimensionality of the affine space/linear
% manifold will be revealed, however, when we "return" this degree of
% freedom by mean-centering in the pre-processing phase of PCA, for in so
% doing we bring the origin back into the manifold, dispensing with the
% extra dimension that arose to accomodate it in the erstwhile vector
% space.
if nvp.affine
    vectors = vectors * nvp.affine(1) + nvp.affine(2); 
end

% Add Gaussian noise to each row (note the transposes) independently.
vectors = vectors + randn(size(vectors'))'*nvp.noiseFactor;
    
end


function vectors = constructLowRankMatrix2(lowRank, nRows, nCols, nvp)
% Uses random linear combinations of orthonormal basis vectors. Watch out
% that "basis" is translated away from origin before generating data, so
% the true dimensionality of the manifold here will be one less than the
% specified 'lowRank'. May come back to address this, but I think this
% method is inferior to the SVD one above anyhow.

arguments
    % number of vectors in basis for row space prior to addition of
    % noise.
    lowRank {mustBeInteger} 
    nRows {mustBeInteger}
    nCols {mustBeInteger}
    nvp.basis = 'orth' % Method used to generate basis for row space.
    nvp.noiseFactor = 1
    % If this value > 1, value will be the scalar to which weights of
    % linear combination must sum.
    nvp.weightConstraint = 0 
end
    
% Create sequence of n linearly independent vectors, where n = lowRank.
switch nvp.basis 
    case 'orth'
        basis = orth(randn(nCols, lowRank))' * 30 + 11;
    case 'gramschmidt'
        basis = gramschmidt(randn(nCols, lowRank))' * 30 + 11;
    case 'randn'
        basis = randn(lowRank, nCols) * 2 + 11;
end

% Generate each row of data matrix by taking random linear combination of
% basis.
vectors = zeros(nRows, nCols);
for iRow = 1:nRows
    % Prevent enormously large weights arising by chance with while
    % loop constraint. First initialize weights to some number with abs
    % magnitude greater than 1 to enter loop.
    weights = 3;
    while sum(abs(weights) > 2) ~= 0
        weights = randn(lowRank, 1);
        % Constrain weights in linear combination so that they add to
        % specified value if desired.
        if nvp.weightConstraint > 0
            weights = weights / (sum(weights)/nvp.weightConstraint);
        end
    end
    vectors(iRow,:) = sum(bsxfun(@times, basis, weights));
end

% Add Gaussian noise.
vectors = vectors + randn(size(vectors))*nvp.noiseFactor;
    
end


function plotSimData(origVecs, pcaOut, plotAs, pcaVecType)

arguments 
    origVecs % Original data prior to PCA.
    pcaOut % Structure containing PCA results.
    plotAs % Controls plotting behavior. May be 'points' or 'vectors'.
    pcaVecType
end

% Plot heatmap of matrix.
close all
subplot(1,2,1)
imagesc(origVecs)
xlabel('Neurons')
ylabel('Conditions')
title('Neural Population')
h = colorbar;
ylabel(h, 'FR (Hz)')

% Change 'prinComps' (consistent among my scripts) to 'principal
% components' for plotting purposes.
if strcmp(pcaVecType, 'prinComps')
    pcaVecType = 'principal components';
end

% Plot loading vectors.
switch pcaVecType
    case 'loadings'
        pcaVecs = pcaOut.loadings;
    case 'eigenvectors'
        pcaVecs = pcaOut.V;
    case 'principal components'
        pcaVecs = pcaOut.prinComps;
    case 'left singular vectors'
        pcaVecs = pcaOut.U;
    otherwise 
        error('Invalid or missing value for pcaVecType.')
end
subplot(1,2,2)
switch plotAs
    case 'points'
        if size(pcaVecs, 2) >=3
            scatter3(pcaVecs(:,1), pcaVecs(:,2),...
                     pcaVecs(:,3))
        elseif size(pcaVecs, 2) == 2
            scatter(pcaVecs(:,1), pcaVecs(:,2))
        elseif size(pcaVecs, 2) == 1
            warning('No plot produced as dimensionality of loadings is 1.')
        end       
    case 'lines'
        if size(pcaVecs, 2) >=3
            plot_vectors(pcaVecs, 3, []) % third arg is color, leaving empty uses default blue
        elseif size(pcaVecs, 2) == 2
            plot_vectors(pcaVecs, 2, [])
        elseif size(pcaVecs, 2) == 1
            warning('No plot produced as dimensionality of loadings is 1.')
        end
end
title(sprintf('%s', pcaVecType))
xlabel('PC 1')
ylabel('PC 2')
zlabel('PC 3')
% xlim([-1.15 1.15])
% ylim([-1.15 1.15])
% zlim([-1.15 1.15])

end

% function vectors = constructAntiCorrNeuronClusters(nNeurons, nClusters,...
%                                                    nCond, nvp)
% arguments
%     nNeurons {mustBeInteger} % number of neurons per cluster
%     nClusters {mustBeEven} % number of clusters must be even positive int
%     nCond {mustBeInteger} % number of conditions
%     nvp.firingRateRange = [2 20]
%     nvp.construction = 'randn' 
%     nvp.noiseFactor = ones(20,1)
% end
% 
% assert(length(nvp.noiseFactor) >= nClusters,...
%        "Length of 'noiseFactor' must be >= nClusters.")
% 
% seedPattern = randi([0 1],nCond,1)*5+5;
% vectors = [];
% for i = 1:nClusters
%     if i == 1
%         vectors = seedPattern
%     vectors = [vectors newCluster];
%     
%                                                
%                                                
% end
% 
% end