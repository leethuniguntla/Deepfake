% Load your trained model
load('deepfake_model.mat', 'netTransfer');  % Replace with 'net' if you used that

% Create a datastore pointing to the folder
trainFolder = fullfile(pwd, 'Final Dataset');

% Create datastore from the training folder
imdsTest = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
% Resize all images to match input size
augTest = augmentedImageDatastore([224 224], imdsTest);

% Classify all images
[YPred, scores] = classify(netTransfer, augTest);

% Display predictions with filenames
for i = 1:length(imdsTest.Files)
    [~, name, ext] = fileparts(imdsTest.Files{i});
    fprintf("🖼️ %s%s → Prediction: %s (Confidence: %.2f%%)\n", ...
        name, ext, string(YPred(i)), max(scores(i, :)) * 100);
end
