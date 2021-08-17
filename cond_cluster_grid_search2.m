function stats = cond_cluster_grid_search2(clustNums,clustSizes,nNeurons,nvp)
% Creates a series of datsets by varying both the number and cardinality of
% gaussian clusters and tests cluster tendency on each dataset. Then,
% regresses metrics of cluster tendency (ePAIRS PAIRS statistic and p
% value) against the hyperparameters used to construct the datasets and
% returns these results. Also plots regression results.
%
% PARAMETERS
% ----------
% clustNums  -- Vector of cluster numbers (number of clusters in each
%              datset).
% clustSizes -- Vector of cluster sizes/cardinalities. 
% nNeurons   -- Number of neurons for an nConditions x nNeurons matrix (the
%               dimensionality of each condition observation).
% Name-Value Pairs (nvp)
%   'noiseFactor'     -- Scalar value that controls the noise added to each
%                        cluster in each dataset. This factor is equivalent
%                        to the variance of the gaussian modeling the
%                        noise.
%   'nReps'           -- Number of datasets to generate for each
%                        hyperparameter pair. Results are averaged across
%                        datasets (with correction factor for p values).
%   'test'            -- String value, either 'elliptical' or 'spherical',
%                        corresponding to the ePAIRS parameters.
%   'kNN'             -- Scalar value that is the number of nearest
%                        neighbors to use in ePAIRS. Default is three.
%   'prior'           -- String value, either 'gaussian' or 'permutation',
%                        specifying prior for ePAIRS. Default is
%                        'gaussian', also see cluster_demo in Tests_demos
%                        directory.
%   'nBootstraps'     -- Scalar value specifying number of null
%                        datasets to be used in ePAIRS' Monte Carlo
%                        simulations.
%   'PEV'             -- Scalar value specifying the proportion of
%                        that the retained principal components should
%                        explain.
%   'pVal'            -- String value specifying sidedness of p value in
%                        ePAIRS. Options are 'left', 'right', and 'both'
%                        (default).
%   'correction'      -- Specify whether to apply a correction to the p
%                        p values if averaged over datasets. Options are
%                        'times 2' and 'ln K' (note the caps). If no
%                        repeated datasets are used (i.e., 'nReps' = 1), it
%                        is advisable to set 'correction' to false or 0.
%   'distr'           -- String value specifying distribution of response
%                        variable for the GLM. Default is 'normal', which
%                        will cause the link function (under the default
%                        settings of glmfit.m) to be the identity, with
%                        f(u) = u.
%   'interactionTerm' -- Logical true or false specifying whether or not to
%                        include a nonlinear interaction term between the
%                        two hyperparameters in the regression models.
%                        Default false.
%   'surfPlot'        -- Option to return surface plot ePAIRS PAIRS stat
%                        and p values as function of hyperparamters.
%   'regressPlot'     -- Option to plot ePAIRS PAIRS stat and p values as
%                        points, with the fitted regression plane displayed
%                        as well (this is the reconstruction of the reponse
%                        variable in row space, not to be confused with the
%                        vector projection of the response variable into
%                        the column space of the design matrix.
%
% RETURNS
% stats -- 1 x 2 cell array (one cell each for the PAIRS stat and ePAIRS p
%          value) whose cells contain a struct containing the default
%          fields of the stats variable returned by MATLAB's glmfit.m
%          function, with the following fields added:
%   .vif        -- Vector of variance inflation factors for each of the
%                  predictors (length = nPredictors, not including
%                  intercept).
%   .cd         -- Scalar value that is the R^2 for the regression model.
%   .cdAdj      -- Scalar value that is the R^2 for the regression model,
%                  adjusted for number of predictors.
%   .cpd        -- Vector (of length nPredictors, not including intercept),
%                  of coefficients of partial determination, one for each
%                  predictor in the regression model.
%   .regressand -- String value corresponding to the regressand for the
%                  current model (either the PAIRS stat or the ePAIRS p
%                  value). For housekeeping purposes.
%
% Author: Jonathan Chien 7/3/21.


arguments
   clustNums
   clustSizes
   nNeurons
   nvp.noiseFactor = 1
   nvp.nReps = 10
   nvp.test = 'elliptical'
   nvp.kNN = 3
   nvp.prior = 'gaussian' 
   nvp.nBootstraps = 1000
   nvp.PEV = 0.8
   nvp.pVal = 'both'
   nvp.correction = false 
   nvp.distr = 'normal'
   nvp.interactionTerm = false 
   nvp.surfPlot = true
   nvp.regressPlot = true
end


%% Preprocessing and ePAIRS/PAIRS

% Enforce column vectors (important for repmat operations in construction
% of design matrix).
if isrow(clustNums), clustNums = clustNums'; end
if isrow(clustSizes), clustSizes = clustSizes'; end

% Determine number of values for each hyperparameter. Also assign nvp.nReps
% to a variable, since error seems to be thrown if nvp.nReps used directly
% in loop definition line?
nClustNums = length(clustNums);
nClustSizes = length(clustSizes);
nReps = nvp.nReps;

% Preallocate.
PAIRSstats = NaN(nClustNums, nClustSizes, nvp.nReps);
pValues = NaN(nClustNums, nClustSizes, nvp.nReps);

% Try all combinations of hyperparameters.
parfor iClustNum = 1:nClustNums
    for iClustSize = 1:nClustSizes
        for iRep = 1:nReps
        
            % Generate clusters of conditions across neurons. 
            CxN = construct_clusters(clustNums(iClustNum), ...
                                     clustSizes(iClustSize), ...
                                     nNeurons, ...
                                     'noiseFactor', ...
                                     nvp.noiseFactor*ones(clustNums(iClustNum),1), ...
                                     'plot', false);  

            % Perform PCA.
            pcaOut = pca_jmc(CxN, 'corrMatrix', true, 'PEV', nvp.PEV);

            % Test for cluster tendency.
            params = [];
            params.nKNN = nvp.kNN; 
            params.prior = nvp.prior; 
            params.origVectors = CxN;
            params.nBootstraps = nvp.nBootstraps;
            params.test_side = nvp.pVal
            switch nvp.test
                case 'elliptical'
                    params.dimSizes = pcaOut.eigenvalues;
                    [PAIRSstats(iClustNum,iClustSize,iRep), ...
                     pValues(iClustNum,iClustSize,iRep), ~] ...
                        = pairsClusterTest_elliptical(pcaOut.loadings, params);
                case 'spherical'
                    [PAIRSstats(iClustNum,iClustSize,iRep), ...
                     pValues(iClustNum,iClustSize,iRep), ~] ...
                        = pairsClusterTest_elliptical(pcaOut.V, params);
            end
        end
    end
end

% Average across repetitions.
PAIRSstats = mean(PAIRSstats, 3);
pValues = mean(pValues, 3);

% Option to apply correction to p values.
if nvp.correction
    switch nvp.correction
        case 'times 2'
            pValues = pValues * 2;
            
        case 'ln K'
            pValues = pValues * log(nvp.nReps);
            if nvp.nReps == 1
                % Warn if K < 3, as ln(K) = 0 in this case.
                warning(['nReps is 1, meaning that the correction factor on ' ...
                         'the p values will be 0 (all p values will equal 0). ' ...
                         'Consider increasing nReps, using a correction factor '
                         'of *2 instead of ln K, or not setting a correction factor.'])
            elseif nvp.nReps < 3
                % Warn if K < 3, as ln(K) < 1 in this case.
                warning(['nReps is less than 3, and meaning that the correction ' ...
                         'factor on the p value will be less than 1. Consider ' ...
                         'increasing nReps, using a correction factor of *2 ' ...
                         'instead of ln K, or not setting a correction factor.'])
            end
    end
end


%% Regression

% Construct design matrix for regression with each hyperparameter as a
% predictor. MATLAB's glmfit asks user to omit constant term in the input
% design matrix.
designMat(:,1) = repelem(clustNums, nClustSizes);
designMat(:,2) = repmat(clustSizes, nClustNums, 1);
if nvp.interactionTerm
    designMat(:,3) = designMat(:,1) .* designMat(:,2);
end
                          
% Construct response vectors by unwinding the arrays of PAIRS stats and p
% values. Must transpose or order will be scrambled (we need elements of
% unwound vector to correspond to going across the rows of the matrix
% here, though they correspond to going down columns by default).
regressand1 = reshape(PAIRSstats', [nClustNums*nClustSizes 1]);
regressand2 = reshape(pValues', [nClustNums*nClustSizes 1]);
                  
% Fit regression model.
[~,~,stats{1}] = glmfit(designMat, regressand1, nvp.distr);
[~,~,stats{2}] = glmfit(designMat, regressand2, nvp.distr);

% Calculate VIF (variance inflation factor), same for both models, as they
% share the same design matrix. 
stats{1}.vif = vif(designMat);
stats{2}.vif = vif(designMat);

% Calculate coefficient of determination (R^2), adjusted R^2, and
% coefficient of partial determination.
[stats{1}.cd, stats{1}.cdAdj, stats{1}.cpd] = CD_CPD(designMat, regressand1);
[stats{2}.cd, stats{2}.cdAdj, stats{2}.cpd] = CD_CPD(designMat, regressand2);

% Add regressand name as field of each struct for reference.
stats{1}.regressand = 'PAIRS statistic';
stats{2}.regressand = 'PAIRS p value';


%% Plotting

% Option to plot p values and PAIRS stat as function of hyperparameters.
if nvp.surfPlot
    % Get grid of hyperparameter values.
    [X,Y] = meshgrid(clustSizes, clustNums);
    
    figure
    
    % Plot PAIRS stats.
    subplot(1,2,1)
    surf(X, Y, PAIRSstats, 'Edgecolor', 'interp')
    title('PAIRS statistic')
    xlabel('Cluster size')
    ylabel('Cluster number')
    zlabel('PAIRS statistic')
    
    % Plot p values.
    subplot(1,2,2)
    surf(X, Y, pValues, 'Edgecolor', 'interp')
    title('PAIRS p value')
    xlabel('Cluster size')
    ylabel('Cluster number')
    zlabel('PAIRS p value')
    
    sgtitle('Clustering behavior over various cluster number and sizes')
end

if nvp.regressPlot
    % y_hat (predicted value for regressand) is the projection of the
    % response vector y (or b in the canonical A*x_hat = P*b) into the
    % column space of the design matrix, A. This projection is easily found
    % by taking a linear combination of the columns of A, with the weights
    % being the beta coefficients.
    y_hat_PAIRSstat = [ones(size(designMat,1),1) designMat] ...
                      * stats{1}.beta;
    y_hat_pValues = [ones(size(designMat,1),1) designMat] ...
                    * stats{2}.beta;
               
    % Reshape y_hat (predicted regressand) back into a nClustNums x
    % nClustSizes matrix. Note the transpose (again linear indices go down
    % the columns, but we want them across the rows, since we hold
    % clustNums(1) constant in the first row and iterate through all
    % clustSizes before moving to the next row, which has clustNums(2)
    % etc.).
    y_hat_PAIRSstat_mat = reshape(y_hat_PAIRSstat, nClustSizes, nClustNums)';
    y_hat_pValues_mat = reshape(y_hat_pValues, nClustSizes, nClustNums)';
    
    % Get grid of hyperparameter values.
    [X,Y] = meshgrid(clustSizes, clustNums);
    
    % Plot data with regression plane (which is NOT the column space of the
    % design matrix. We are in row space here). First, for the PAIRSstat.
    figure
    subplot(1,2,1)
    surf(X, Y, y_hat_PAIRSstat_mat)
    alpha 0.5
    hold on
    plot3(designMat(:,2), designMat(:,1), regressand1, ...
          'o', 'MarkerFaceColor', [0 0.4470 0.7410])
    hold off
    title('PAIRS statistic')
    xlabel('Cluster cardinality')
    ylabel('Number of clusters')
    zlabel('PAIRS statistic')
    
    % Now for the p values.
    subplot(1,2,2)
    surf(X, Y, y_hat_pValues_mat)
    alpha 0.5
    hold on
    plot3(designMat(:,2), designMat(:,1), regressand2, ...
          'o', 'MarkerFaceColor', [0 0.4470 0.7410]) 
    hold off
    title('PAIRS p value')
    xlabel('Cluster size')
    ylabel('Cluster number')
    zlabel('PAIRS p value')
    
    sgtitle('OLS fit to clustering behavior over various cluster number and sizes')
end

end