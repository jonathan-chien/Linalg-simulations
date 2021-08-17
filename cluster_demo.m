%%%%%%%%%%%%%%%%% Modified Version of make_clusters script %%%%%%%%%%%%%%%%

%% Case 1 
% Homoscedastic clusters with equal cardinality and centroids offset only
% along x axis.

C1 = randn(100,2);
C2 = randn(100,2);
C2(:,1) = C2(:,1) + 10; % shift one of the clusters along x axis
shuffle = 'both';

X = [C1; C2];
X_perm = NaN(size(X));
switch shuffle
    case 'columns'
        for i = 1:size(X,2)
            X_perm(:,i) = X(randperm(size(X,1)),i);
        end
    case 'rows'
        for i = 1:size(X,1)
            X_perm(i,:) = X(i,randperm(size(X,2)));
        end
    case 'both' % shuffle columns, then rows
        for i = 1:size(X,2)
            X_perm(:,i) = X(randperm(size(X,1)),i);
        end
        for i = 1:size(X,1)
            X_perm(i,:) = X(i,randperm(size(X,2)));
        end
end

figure
subplot(2,1,1)
plot(X(:,1), X(:,2), 'o')
title('Clusters (data)')

subplot(2,1,2)
plot(X_perm(:,1), X_perm(:,2), 'o')
title('Null from shuffling')

% If a region is sparse along both dimensions, e.g., the center of the
% graph, that region will remain sparse even after shuffling, because in
% order for there to be data points there, there must be x or y coordinates
% in that region that can be "shuffled in," which there are not. Regions of
% density vs sparsity in the shuffled data are determined by the original
% data, and the shuffled data is only random up to those constraints
% determined by the particular structure of the empirical data. One way of
% visualizing this sparseness/density is by unwinding the matrix into a
% single vector and plotting a histogram of its elements to visualize
% density/sparsity of the matrix elements along both dimensions.
figure
histogram(X(:),30);
title('Element density')


%% Case 2
% Homoscedastic clusters with equal cardinality and centroids offset along
% both x and y axis.

C1 = randn(100,2);
C2 = randn(100,2) + 10; % shift cluster 2 wrt cluster 1 along both axes

X = [C1; C2];
X_perm = NaN(size(X));
for i = 1:size(X,2)
    X_perm(:,i) = X(randperm(size(X,1)),i);
end

figure
subplot(2,1,1)
plot(X(:,1), X(:,2), 'o')
title('Clusters (data)')

subplot(2,1,2)
plot(X_perm(:,1), X_perm(:,2), 'o')
title('Null from shuffling')

figure
histogram(X(:),30);
title('Element density')


%% Case 3
% Non-homoscedastic clusters with equal cardinality and centroids offset
% along only x axis.

C1 = randn(100,2);
C2 = randn(100,2)*4;
C2(:,1) = C2(:,1) + 10; % shift one of the clusters along x axis

X = [C1; C2];
X_perm = NaN(size(X));
for i = 1:size(X,2)
    X_perm(:,i) = X(randperm(size(X,1)),i);
end

figure
subplot(2,1,1)
plot(X(:,1), X(:,2), 'o')
title('Clusters (data)')

subplot(2,1,2)
plot(X_perm(:,1), X_perm(:,2), 'o')
title('Null from shuffling')

figure
histogram(X(:),30);
title('Element density')


%% Case 4
% Homoscedastic clusters with equal cardinality and centroids offset only
% along x axis, with gaussian "noise" in between them (essentially a larger
% cluster with centroid at the midpoint of the tighter clusters).

C1 = randn(100,2);
C2 = randn(100,2);
C2(:,1) = C2(:,1) + 10; % shift one of the clusters along x axis
N = randn(100,2)*4;
N(:,1) = N(:,1) + 5;

X = [C1; C2; N];

X_perm = NaN(size(X));
for i = 1:size(X,2)
    X_perm(:,i) = X(randperm(size(X,1)),i);
end

figure
subplot(2,1,1)
plot(X(:,1), X(:,2), 'o')
title('Clusters (data)')

subplot(2,1,2)
plot(X_perm(:,1), X_perm(:,2), 'o')
title('Null from shuffling')

figure
histogram(X(:),30);
title('Element density')


%% Case 5
% Homoscedastic clusters with equal cardinality and centroids offset along
% both x and y axis, with gaussian "noise" in between them (essentially a
% larger cluster with centroid at the midpoint of the tighter clusters).

C1 = randn(100,2);
C2 = randn(100,2) + 10; % shift one of the clusters along x axis
N = randn(100,2)*4 + 5; % noise is in between

X = [C1; C2; N];

X_perm = NaN(size(X));
for i = 1:size(X,2)
    X_perm(:,i) = X(randperm(size(X,1)),i);
end

figure
subplot(2,1,1)
plot(X(:,1), X(:,2), 'o')
title('Clusters (data)')

subplot(2,1,2)
plot(X_perm(:,1), X_perm(:,2), 'o')
title('Null from shuffling')

figure
histogram(X(:),30);
title('Element density')


%% Case 6
% Non-homoscedastic clusters with unequal cardinality and centroids offset
% along both x and y axis, with gaussian "noise" in between them
% (essentially a larger cluster with centroid at the midpoint of the
% tighter clusters).

C1 = randn(100,2)*2.5;
C2 = randn(70,2) + 10; % shift one of the clusters along x axis
N = randn(100,2)*7 + 7; % noise is in between

X = [C1; C2; N];

X_perm = NaN(size(X));
for i = 1:size(X,2)
    X_perm(:,i) = X(randperm(size(X,1)),i);
end

figure
subplot(2,1,1)
plot(X(:,1), X(:,2), 'o')
title('Clusters (data)')

subplot(2,1,2)
plot(X_perm(:,1), X_perm(:,2), 'o')
title('Null from shuffling')

figure
histogram(X(:),30);
title('Element density')


%% Case 7
% Homoscedastic clusters with equal cardinality and centroids offset along
% both x and y axis, with gaussian "noise" in between them (essentially a
% larger cluster with centroid at the midpoint of the tighter clusters).
% This time we are shuffling rows and not columns.

C1 = randn(100,2);
C2 = randn(100,2) + 10; % shift one of the clusters along x axis
N = randn(100,2)*4 + 5; % noise is in between

X = [C1; C2; N];

X_perm = NaN(size(X));
for i = 1:size(X,1)
    X_perm(i,:) = X(i,randperm(size(X,2)));
end

figure
subplot(2,1,1)
plot(X(:,1), X(:,2), 'o')
title('Clusters (data)')

subplot(2,1,2)
plot(X_perm(:,1), X_perm(:,2), 'o')
title('Null from shuffling')

figure
histogram(X(:),30);
title('Element density')


%% Case 8
% 3 homoscedastic clusters with equal cardinality in 3D space.

C1 = randn(100,3);
C2 = randn(100,3);
C2(:,1) = C2(:,1) + 10; % shift one of the clusters along x axis
C3 = randn(100,3)+ 20;

X = [C1; C2; C3];
X_perm = NaN(size(X));
for i = 1:size(X,2)
    X_perm(:,i) = X(randperm(size(X,1)),i);
end

figure
subplot(2,1,1)
plot3(X(:,1), X(:,2), X(:,3), 'o')
title('Clusters (data)')
grid on

subplot(2,1,2)
plot3(X_perm(:,1), X_perm(:,2), X_perm(:,3), 'o')
title('Null from shuffling')
grid on

figure
histogram(X(:),30);
title('Element density')


%% Case 8a
% Only one cluster in 3D space.

X = repmat([0 10 0], 100, 1);
X = X + randn(size(X))*0.15;

X_perm = NaN(size(X));
for i = 1:size(X,2)
    X_perm(:,i) = X(randperm(size(X,1)),i);
end

figure

subplot(2,1,1)
plot3(X(:,1), X(:,2), X(:,3), 'o')
title('Clusters (data)')
subplot(2,1,1)
grid on

subplot(2,1,2)
plot3(X_perm(:,1), X_perm(:,2), X_perm(:,3), 'o')
title('Null from shuffling')
subplot(2,1,2)
grid on

figure
histogram(X(:),30);
title('Element density')


%% Case 9
% 3 homoscedastic clusters with equal cardinality in 3D space. This time
% shuffling within each row rather than within columns.

C1 = randn(100,3);
C2 = randn(100,3);
C2(:,1) = C2(:,1) + 10; % shift one of the clusters along x axis
C3 = randn(100,3)+ 20;

X = [C1; C2; C3];
X_perm = NaN(size(X));
for i = 1:size(X,1)
    X_perm(i,:) = X(i,randperm(size(X,2)));
end

figure
subplot(2,1,1)
plot3(X(:,1), X(:,2), X(:,3), 'o')
title('Clusters (data)')
grid on

subplot(2,1,2)
plot3(X_perm(:,1), X_perm(:,2), X_perm(:,3), 'o')
title('Null from shuffling')
grid on

figure
histogram(X(:),30);
title('Element density')


%% Case 10

C1 = randn(100,2)*4;
C2 = randn(100,2);
C2(:,1) = C2(:,1)+4;

X = [C1; C2];
X_perm = NaN(size(X));
for i = 1:size(X,2)
    X_perm(:,i) = X(randperm(size(X,1)),i);
end

figure
subplot(2,1,1)
plot(X(:,1), X(:,2), 'o')
title('Clusters (data)')

subplot(2,1,2)
plot(X_perm(:,1), X_perm(:,2), 'o')
title('Null from shuffling')

figure
histogram(X(:),30);
title('Element density')


%% Case 11: one elongated blob in 2D (2 conditions)

clearvars
close all
nNeurons = 200;

% Gaussian noise sphere.
A = [randn(1, nNeurons); randn(1, nNeurons)];

% Strech along second dimension.
A(2,:) = A(2,:)*2;

% Rotate.
X = [ 1/sqrt(2) 1/sqrt(2);
     -1/sqrt(2) 1/sqrt(2)];            
A = X*A;

maxCoord = max(abs(A), [], 'all');

% Shuffle conditions for each neuron
for i = 1:nNeurons
    A_perm(:,i) = A(randperm(2),i);
end


% Plot
figure
subplot(1,2,1)
scatter(A(1,:), A(2,:))
grid on
xlabel('Condition 1')
ylabel('Condition 2')
xlim([-maxCoord*1.15 maxCoord*1.15])
ylim([-maxCoord*1.15 maxCoord*1.15])

subplot(1,2,2)
scatter(A_perm(1,:), A_perm(2,:))
grid on
xlabel('Condition 1')
ylabel('Condition 2')
xlim([-maxCoord*1.15 maxCoord*1.15])
ylim([-maxCoord*1.15 maxCoord*1.15])

% Visualize sparsity/density along both dimensions.
figure
histogram(A(:),30)
title('Element density')

%% Case 12: one elongated blob in 3D (3 conditions)

clearvars
close all
nNeurons = 400;
plotAs = 'points';

% Gaussian noise sphere.
A = [randn(1, nNeurons); randn(1, nNeurons); randn(1, nNeurons)];

% Strech along first dimension.
A(1,:) = A(1,:)*5;

% Rotate.
X = [1/sqrt(2) -1/sqrt(2) -1/sqrt(2);
     1/sqrt(2)  1/sqrt(2) -1/sqrt(2);
     1/sqrt(2)      0      1/sqrt(2)];
A = X*A;

maxCoord = max(A, [], 'all');


% Shuffle conditions for each neuron
for i = 1:nNeurons
    A_perm(:,i) = A(randperm(3),i);
end

switch plotAs
    case 'points'
        subplot(1,2,1)
        scatter3(A(1,:), A(2,:), A(3,:))
        xlabel('Condition 1')
        ylabel('Condition 2')
        zlabel('Condition 3')
        xlim([-maxCoord maxCoord])
        ylim([-maxCoord maxCoord])
        zlim([-maxCoord maxCoord])
        title('Original data')

        subplot(1,2,2)
        scatter3(A_perm(1,:), A_perm(2,:), A_perm(3,:))
        xlabel('Condition 1')
        ylabel('Condition 2')
        zlabel('Condition 3')
        xlim([-maxCoord maxCoord])
        ylim([-maxCoord maxCoord])
        zlim([-maxCoord maxCoord])
        title('Permuted data')
    case 'vectors'
        subplot(1,2,1)
        plot_vectors(A', 3)
        xlabel('Condition 1')
        ylabel('Condition 2')
        zlabel('Condition 3')
        xlim([-maxCoord maxCoord])
        ylim([-maxCoord maxCoord])
        zlim([-maxCoord maxCoord])
        title('Original data')

        subplot(1,2,2)
        plot_vectors(A_perm', 3)
        xlabel('Condition 1')
        ylabel('Condition 2')
        zlabel('Condition 3')
        xlim([-maxCoord maxCoord])
        ylim([-maxCoord maxCoord])
        zlim([-maxCoord maxCoord])
        title('Permuted data')
end

figure
histogram(A(:),30)
title('Element density')

