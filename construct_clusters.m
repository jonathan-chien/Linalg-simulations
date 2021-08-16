function vectors = construct_clusters(nClusters,card,nDims,nvp)
% Creates m clusters of cardinality c in R^n. Data points may be placed
% either as rows or columns of matrix.
%
% PARAMETERS
% ----------
% nClusters -- Number of clusters.
% card      -- Cardinality of each cluster (number of points in each
%              cluster). Future versions of this function may allow
%              different cluster sizes.
% nDims     -- Dimensionality of the data. nDims = n, for clustered data in
%              R^n.
% Name-Value Pairs (nvp)
%   'construction' -- String value, either 'randn' or 'randi'. Determines
%                     whether each data point is a random normal vector or
%                     a set of integers drawn from the range specified
%                     through the 'randiRange' name value pair.
%   'translate'    -- If 'construction' = 'randn', this parameter is a
%                     2-vector allowing the clusters to be scaled by a
%                     factor (first element of 'translate') and translated
%                     along each dimension by a displacement (second
%                     element of 'translate' is the componenet of the
%                     dispalcement vector (identical) along each
%                     dimension). [2 11] usually works well for creaing
%                     biologically plausible synthethic firing rate data
%                     and is the default. If this behavior is not desired,
%                     set 'translate' to false or an empty vector.
%   'randiRange'   -- If 'construction' is 'randi', supply a 2-vector whose
%                     first and second elements are respectively the upper
%                     and lower limits of the range of integers from which
%                     to draw values for the components data points.
%   'noiseFactor'  -- n-vector, where n >= the number of clusters. The i_th
%                     element is a scalar factor multiplying the variance
%                     of the gaussian matrix overlaid on top of the i_th
%                     cluster. If a given such element is zero, then all
%                     the points in the cluster will be at the exact same
%                     location in R^n.
%   'clustersIn'   -- String value, either 'rows' or 'columns'. If 'rows',
%                     clusters appear amongst the row vectors; if
%                     'columns', the clusters appear amongst the column
%                     vectors.
%   'plot'         -- Logical true (default) or false. If true, plots the
%                     data matrix as a heatmap. May be desirable to
%                     suppress this behavior if this function is called
%                     repeatedly in Monte Carlo simulations etc.
%   'subplot'      -- Logical true or false (default). If false, a new
%                     figure is generated, onto whose axes the heatmap will
%                     be plotted. If true, generation of a new figure will
%                     be suppressed, and the heatmap will be plotted onto
%                     the current axes (e.g., a subplot within a function
%                     calling this one).           
%   'colormap'     -- Specify a colormap, corresponding to options within
%                     MATLAB's colormap function. Some options include
%                     'cool', 'autumn', 'spring', 'turbo', and 'paruala'.
%
% RETURNS
% -------
% vectors -- m*c x n matrix consisting by default of m clusters, each with
%            c points, in R^n. May optionally be transposed as well, so
%            that clusters are amongst the column vectors.
% heatmap -- Optional heatmap of vectors.
%
% Author: Jonathan Chien Version 1.0 Created 7/11/21. Last edit: 7/12/21.
% Version history:
%   -- Adapted from a local function in conditions_vs_neurons script in the
%      Tests_demos directory.

arguments
    nClusters (1,1) {mustBeInteger}
    card (1,1) {mustBeInteger}
    nDims (1,1) {mustBeInteger}
    nvp.construction = 'randn' 
    nvp.translate = [2 11];
    nvp.randiRange = [2 18] % only used if 'construction' = randi
    nvp.noiseFactor = ones(20,1)
    nvp.clustersIn = 'rows'
    nvp.plot = true
    nvp.subplot = false
    nvp.colormap = 'cool'
end

assert(length(nvp.noiseFactor) >= nClusters, ...
       "Length of 'noiseFactor' must be >= nClusters.")

% Preallocate.
vectors = zeros(nClusters*card, nDims);

% Generate each cluster as set of repeats of a random normal vector (with
% number of repeats equal to the desired cardinality of each cluster).
for iClust = 1:nClusters
    
    % Construct current cluster of conditions according to specified
    % method.
    switch nvp.construction
        case 'randi'            
            vectors((iClust-1)*card+1 : (iClust-1)*card+card, :)...
                = repmat(randi(nvp.randiRange,1,nDims),card,1);
        case 'randn'
            newClust = repmat(randn(1,nDims),card,1);
            if nvp.translate
                newClust = newClust * nvp.translate(1) + nvp.translate(2);
            end
            vectors((iClust-1)*card+1 : (iClust-1)*card+card, :)...
                = newClust;
    end
    
    % Add gaussian noise to the newly created cluster.
    vectors((iClust-1)*card+1 : (iClust-1)*card+card, :)...
        = vectors((iClust-1)*card+1 : (iClust-1)*card+card, :) +...
          randn(size(vectors((iClust-1)*card+1 :...
                             (iClust-1)*card+card, :))) ...
          *nvp.noiseFactor(iClust);
end

% Option to transpose to place clusters in columns of matrix.
switch nvp.clustersIn
    case 'rows'
        % Do nothing.
    case 'columns'
        vectors = vectors';
    otherwise 
        error("Invalid or missing value for 'clustersIn'.")
end

% Option to plot heatmap of matrix.
if nvp.plot
    % If this function is being called by another function within a
    % subplot, suppress generation of new figure here.
    if ~nvp.subplot
        figure
    end
    colormap(nvp.colormap)
    imagesc(vectors)
    colorbar
end

end
