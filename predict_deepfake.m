function result = predict_deepfake(image_path)
% PREDICT_DEEPFAKE Predicts whether a given image is Real or Deepfake
% Input: image_path - absolute path to image
% Output: result - string "Label (Confidence%)"

    try
        % Load your trained model
        modelPath = fullfile('C:', 'Users', 'DELL', 'my project', ...
                             'deepdetect-chatbot', 'deepfake_model.mat');
        load(modelPath, 'netTransfer');  % Make sure netTransfer is the variable name

        % Read uploaded image
        img = imread(image_path);

        % Convert grayscale to RGB if needed
        if size(img, 3) == 1
            img = cat(3, img, img, img);
        end

        % Resize to match model input
        img = imresize(img, [224 224]);

        % Classify
        [label, scores] = classify(netTransfer, img);
        confidence = max(scores) * 100;

        % Return string result
        result = sprintf('%s (%.2f%%)', string(label), confidence);
        fprintf("✅ %s → %s\n", image_path, result);

    catch ME
        result = sprintf('Error: %s', ME.message);
    end
end
