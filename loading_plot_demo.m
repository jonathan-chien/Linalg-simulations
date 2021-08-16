%% Case 1: Construct matrix A

% Construct a matrix A as a sequence of 5 12-dimensional column vectors
% (i.e., 5 variables). Vars 1 and 2 are correlated, as are vars 3 and 4.
% Var 5 is anticorrelated with vars 3 and 4. Correlated variables have
% small angles between them (0 if maximally correlated, pi if maximally
% anticorrelated). Correlated variables define clusters in the original
% column space. Projections of the variables in the eigenspace are the
% loading vectors, and the cluster structure is denoised, allowing lower
% dimensional representation, but preserved.
clearvars
rng('default')
A = [repmat(randi([0 1],12,1),1,2) repmat(randi([0 1],12,1),1,2)];
A(A(:,4)==0,5) = 1;
A(A(:,4)==1,5) = 0;
A = A + randn(size(A))*0.15;

% Plot heatmap of matrix.
close all
figure
imagesc(A)
xlabel('Variables')
ylabel('Observations')
title('Matrix A')

% Get loading vectors.
pcaOut = pca_jmc(A, 'corrMatrix', true, 'nDims', 3);

% Prepare for plotting.
nVars = size(pcaOut.loadings, 1);

x1 = zeros(1, nVars);
y1 = zeros(1, nVars);
z1 = zeros(1, nVars);

x2 = pcaOut.loadings(:, 1)';
y2 = pcaOut.loadings(:, 2)';
z2 = pcaOut.loadings(:, 3)';

% Plot in 2D.
figure
plot([x1; x2], [y1; y2])
xlabel('PC 1')
ylabel('PC 2')
grid on
title('Loading plot')

% Plot in 3D.
figure
plot3([x1; x2], [y1; y2], [z1; z2])
xlabel('PC 1')
ylabel('PC 2')
zlabel('PC 3')
grid on
title('Loading plot')

%% Case 2: Construct neural response matrix (minimal noise).

% Generate a 12 x 100 matrix representing 12 conditions and 100 neurons,
% with two clusters of neurons across conditions. Reduced noise for better
% visualization.
clearvars
rng('default')
vectors = [repmat(randi([0 1],12,1),1,50) repmat(randi([0 1],12,1),1,50)];
vectors = vectors + randn(size(vectors))*0.25;

% Plot heatmap of neural responses across conditions.
close all
figure
imagesc(vectors)
xlabel('Neurons')
ylabel('Conditions')
title('Data')

% Get loading vectors.
pcaOut = pca_jmc(vectors, 'corrMatrix', true, 'PEV', 1);

% Prepare for plotting.
nNeurons = size(pcaOut.loadings, 1);

x1 = zeros(1, nNeurons);
y1 = zeros(1, nNeurons);
z1 = zeros(1, nNeurons);

x2 = pcaOut.loadings(:, 1)';
y2 = pcaOut.loadings(:, 2)';
z2 = pcaOut.loadings(:, 3)';

% Plot in 2D.
figure
plot([x1; x2], [y1; y2])
xlabel('PC 1')
ylabel('PC 2')
grid on
title('Loading plot')

% Plot in 3D.
figure
plot3([x1; x2], [y1; y2], [z1; z2])
xlabel('PC 1')
ylabel('PC 2')
zlabel('PC 3')
grid on
title('Loading plot')

%% Case 3: 300 neural responses across 3 conditions (low dimensional)

clearvars
rng('default')
nOrigClusters = 2;
noiseFactor = 1;

% Create original data with clustering.
switch nOrigClusters
    case 2
        % Create 3 x 300 matrix with two clusters.
        vectors = [repmat([20; 6; 7],1,150) repmat([9; 16; 18],1,150)];...
        vectors = vectors + randn(size(vectors))*noiseFactor;
    case 3
        % Create 3 x 100 matrix with three clusters.
        vectors = [repmat([20; 6; 7],1,100) repmat([9; 16; 18],1,100)...
                   repmat([2; 2; 12],1,100)]; 
        vectors = vectors + randn(size(vectors))*noiseFactor;
end


% Plot original data in firing rate space.
close all
subplot(2,3,1)
scatter3(vectors(1,:),vectors(2,:),vectors(3,:))
title('Original Data')
xlabel('Condition 1')
ylabel('Condition 2')
zlabel('Condition 3')


% Plot heatmap of original data.
subplot(2,3,2)
imagesc(vectors)
title('Original data')
xlabel('Neurons')
ylabel('Conditions')


% Shuffle original data within columns.
for iNeuron = 1:size(vectors,2)
    vectorsPerm(:,iNeuron) = vectors(randperm(size(vectors,1)),iNeuron);
end

% % Shuffle original data within rows.
% for iCond = 1:size(vectors, 1)
%     vectorsPerm(iCond,:) = vectors(iCond, randperm(size(vectors,2)));
% end

% Plot shuffled data.
switch nOrigClusters
    case 2
        clusterIdx = spectralcluster(vectorsPerm', 6);
    case 3
        clusterIdx = spectralcluster(vectorsPerm', 9);
end
nShuffledClusters = length(unique(clusterIdx));
colorMap = [0, 0.4470, 0.7410; 0.8500, 0.3250, 0.0980;...
            0.9290, 0.6940, 0.1250; 0.4940, 0.1840, 0.5560;...
            0.4660, 0.6740, 0.1880; 0.3010, 0.7450, 0.9330;
            0.6350, 0.0780, 0.1840; 0 0 1; 1 0 0;];
subplot(2,3,3)
for iCluster = 1:nShuffledClusters
    scatter3(vectorsPerm(1,clusterIdx==iCluster),...
             vectorsPerm(2,clusterIdx==iCluster),...
             vectorsPerm(3,clusterIdx==iCluster),...
             36, colorMap(iCluster,:));
    hold on
end
title('Shuffled Data (Input to PCA)')
xlabel('Condition 1')
ylabel('Condition 2')
zlabel('Condition 3')


% Get loading vectors from unshuffled data.
pcaOutUnperm = pca_jmc(vectors, 'CorrMatrix', true, 'PEV', 1);

% Prepare vector tails.
nNeurons = size(pcaOutUnperm.loadings, 1);

x1 = zeros(1, nNeurons);
y1 = zeros(1, nNeurons);

% Plot unshuffled loadings from unshuffled data.
x2 = pcaOutUnperm.loadings(:, 1)';
y2 = pcaOutUnperm.loadings(:, 2)';

subplot(2,3,4)
plot([x1; x2], [y1; y2], 'Color', [0 0.4470 0.7410])
xlabel('PC 1')
ylabel('PC 2')
xlim([-1.1 1.1])
ylim([-1.1 1.1])
grid on
title('Unshuffled data unshuffled loadings')


% Plot shuffled loadings from unshuffled data.
for iDim = 1:size(pcaOutUnperm.loadings,2)
    loadingsPermAfter(:,iDim)...
        = pcaOutUnperm.loadings(randperm(size(pcaOutUnperm.loadings,1)),iDim);
end

x2 = loadingsPermAfter(:, 1)';
y2 = loadingsPermAfter(:, 2)';

subplot(2,3,5)
plot([x1; x2], [y1; y2], 'Color', [0 0.4470 0.7410])
xlabel('PC 1')
ylabel('PC 2')
xlim([-1.1 1.1])
ylim([-1.1 1.1])
grid on
title('Unshuffled data shuffled loadings')


% Get loading vectors from shuffled data.
pcaOutPerm = pca_jmc(vectorsPerm, 'corrMatrix', true, 'PEV', 1);

% Plot loadings from shuffled data.
x2 = pcaOutPerm.loadings(:, 1)';
y2 = pcaOutPerm.loadings(:, 2)';

subplot(2,3,6)
for iCluster = 1:nShuffledClusters
    plot([x1(clusterIdx==iCluster); x2(clusterIdx==iCluster)],...
         [y1(clusterIdx==iCluster); y2(clusterIdx==iCluster)],...
         'Color', colorMap(iCluster,:))
    hold on
end
xlabel('PC 1')
ylabel('PC 2')
zlabel('PC 3')
xlim([-1.1 1.1])
ylim([-1.1 1.1])
grid on
title('Loadings from shuffled data')


%% Case 4: 150 neural responses across 12 conditions (high dimensional)

% Construct 12 x 120 matrix corresponding to 12 conditions and 120 neurons.
% Neurons fall into two clusters of 50 each, with 50 random neurons.
clearvars
rng('default')
vectors = [repmat(randi([0 1],12,1),1,50) repmat(randi([0 1],12,1),1,50)...
           randn(12,50)];
vectors = vectors + randn(size(vectors))*0.15;


% Plot original data.
close all
subplot(2,3,1)
imagesc(vectors)
xlabel('Neurons')
ylabel('Conditions')
title('Original Data')


% Shuffle data.
for iNeuron = 1:size(vectors,2)
    vectorsPerm(:,iNeuron) = vectors(randperm(size(vectors,1)),iNeuron);
end

% Plot heatmap of shuffled data.
subplot(2,3,2)
imagesc(vectorsPerm)
xlabel('Neurons')
ylabel('Conditions')
title('Shuffled Data')


% Get loading vectors from unshuffled data.
pcaOutUnperm = pca_jmc(vectors, 'corrMatrix', true, 'PEV', 0.8);

% Prepare vector tails.
nNeurons = size(pcaOutUnperm.loadings, 1);

x1 = zeros(1, nNeurons);
y1 = zeros(1, nNeurons);
z1 = zeros(1, nNeurons);

% Plot unshuffled loadings from unshuffled data.
x2 = pcaOutUnperm.loadings(:, 1)';
y2 = pcaOutUnperm.loadings(:, 2)';
z2 = pcaOutUnperm.loadings(:, 3)';

subplot(2,3,4)
plot3([x1; x2], [y1; y2], [z1; z2])
xlabel('PC 1')
ylabel('PC 2')
zlabel('PC 3')
grid on
title('Unshuffled data unshuffled loadings')


% Plot shuffled loadings from unshuffled data.
for iDim = 1:size(pcaOutUnperm.loadings, 2)
    loadingsPermAfter(:,iDim)...
        = pcaOutUnperm.loadings(randperm(size(pcaOutUnperm.loadings,1)),iDim);
end

x2 = loadingsPermAfter(:, 1)';
y2 = loadingsPermAfter(:, 2)';
z2 = loadingsPermAfter(:, 3)';

subplot(2,3,5)
plot3([x1; x2], [y1; y2], [z1; z2])
xlabel('PC 1')
ylabel('PC 2')
zlabel('PC 3')
grid on
title('Shuffled loadings from unshuffled data')


% Get loading vectors from shuffled data.
pcaOutPerm = pca_jmc(vectorsPerm, 'corrMatrix', true, 'PEV', 0.8);

% Plot loadings from shuffled data.
x2 = pcaOutPerm.loadings(:, 1)';
y2 = pcaOutPerm.loadings(:, 2)';
z2 = pcaOutPerm.loadings(:, 3)';

subplot(2,3,6)
plot3([x1; x2], [y1; y2], [z1; z2])
xlabel('PC 1')
ylabel('PC 2')
zlabel('PC 3')
grid on
title('Loadings from shuffled data')

% %% ePAIRS for case 3
% % Make sure it was section 3 that was run immediately before running this
% % section. Can't see much here due to low dimensionality of original data.
% % Need to rescale histogram within ePAIRS.
% 
% % use constraint v'v = lambda
% clear params
% params.dimSizes = pcaOutPerm.eigenvalues;
% params.nKNN = 1; %round(sqrt(size(pcaOutPerm.loadings, 1)));
% params.prior = 'permutation';
% 
% figure
% [PAIRSstat, p, dists]...
%     = pairsClusterTest_elliptical(pcaOutPerm.loadings, params);
% 
% 
% %% ePAIRS for case 4
% % Make sure it was section 4 that was run immediately before running this
% % section.
% 
% % use constraint v'v = lambda
% clear params
% params.dimSizes = pcaOutPerm.eigenvalues;
% params.nKNN = 1; %round(sqrt(size(pcaOutPerm.loadings, 1)));
% params.prior = 'permutation';
% 
% figure
% [PAIRSstat, p, dists]...
%     = pairsClusterTest_elliptical(pcaOutPerm.loadings, params);
% 
% %% 
% 
% rng(seed)
% A = randn(12, 150);
% for i = 1:size(A,2)
%     A_perm(:,i) = A(randperm(size(A,1)),i);
% end
% 
% pcaOut = pca_jmc(A, 'corrMatrix', true, 'PEV', 0.8);
% pcaOutPerm = pca_jmc(A_perm, 'corrMatrix', true, 'PEV', 0.8);
% 
% % use constraint v'v = lambda
% close all
% clear params
% params.dimSizes = pcaOutPerm.eigenvalues;
% params.nKNN = 1; %round(sqrt(size(pcaOut.loadings, 1)));
% params.prior = 'permutation';
% 
% figure
% [PAIRSstat, p, dists]...
%     = pairsClusterTest_elliptical(pcaOutPerm.loadings, params);
