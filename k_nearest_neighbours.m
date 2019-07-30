function [labels] = k_nearest_neighbours(Xtrain,Ytrain,Xtest,K,distFunc)

    % Function to implement the K nearest neighbours algorithm on the given
    % dataset.
    % Usage: labels = k_nearest_neighbours(Xtrain,Ytrain,Xtest,K)
    
    % Xtrain : N x P Matrix of training data, where N is the number of
    %   training examples, and P is the dimensionality (number of features)
    % Ytrain : N x 1 Vector of training labels (0/1)
    % Xtest : M x P Matrix of testing data, where M is the number of
    %   testing examples.
    % K : number of nearest neighbours used to make predictions on the test
    %     dataset. Remember to take care of corner cases.
    % distfunc: distance function to be used - l1, l2, linf.
    % labels : return an M x 1 vector of predicted labels for testing data.
    
    if nargin<5
    distFunc = 'l2';
    end
    
    numTestPoints = size(Xtest, 1);
    numTrainPoints = size(Xtrain, 1);

    % Step 1:  Reshape the N x P training matrix into a 1 X P x N 3-D array
    trainMat = reshape(Xtrain', [1 size(Xtrain,2) numTrainPoints]);
    % Step 2:  Replicate the training array for each test point (1st dim)
    trainCompareMat = repmat(trainMat, [numTestPoints 1 1]);
    % Step 3:  Replicate the test array for each training point (3rd dim)
    testCompareMat = repmat(Xtest, [1 1 numTrainPoints]);
    % Step 4:  Element-wise subtraction
    diffMat = testCompareMat - trainCompareMat;

    if strcmp(distFunc, 'l2')
        distMat = sqrt(sum(diffMat.^2, 2));
    elseif strcmp(distFunc, 'l1')
        distMat = sum(abs(diffMat), 2);
    elseif strcmp(distFunc, 'linf')
        distMat = max(abs(diffMat), [], 2);
    else
        error('Unrecognized distance function');
    end

    distMat = squeeze(distMat);
    if numTestPoints == 1
        distMat = distMat';
    end

    [sorted nnMat] = sort(distMat, 2);

    nnMat = nnMat(:,1:K);

    % Average over the nearest neighbors to get predicted labels
    labels = Ytrain(nnMat);
    if size(nnMat, 1) == 1 
        labels = labels';
    end

    if K > 1
        labels = mean(labels, 2);
    end
    
    labels(labels>=0.5)=1;
    labels(labels<0.5)=0;
end