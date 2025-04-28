## Convolutional Neural Networks (CNNs) Recap

**Key Concepts:**
- CNNs use the convolution operator to transform feature maps, sliding a window over the input and computing inner products to extract features.
- Lower layers in CNNs typically learn basic features like edges and colors, while higher layers capture complex object parts.
- Pooling layers are used to downsample and reduce the dimensionality of feature maps.
- Several architectures have shaped the field:
  - **LeNet (1998):** Early 5-layer CNN for digit recognition.
  - **AlexNet (2012):** Sparked the deep learning boom by winning ImageNet with an 8-layer network.
  - **ZFNet (2013):** Similar to AlexNet, won ImageNet classification.
  - **GoogLeNet & VGG (2014):** Much deeper networks, significantly improving classification accuracy.
  - **ResNet (2015):** Introduced extremely deep architectures (up to 150 layers), further improving performance.

## Beyond Classification: Localization and Detection

**Classification:** Assigns a single category label to an image.

**Other Tasks:**
- **Classification + Localization:** Assigns a category label and draws a bounding box around the object of interest.
- **Detection:** Finds all instances of specified object categories in an image, drawing bounding boxes around each.
- **Instance Segmentation:** Identifies all instances of categories and outlines the precise pixels belonging to each instance (not covered in detail in this lecture).

**Key Differences:**
- **Localization:** Typically involves finding a fixed or single object per image.
- **Detection:** Involves finding a variable number of objects per image.

## Classification and Localization

**Summary:**
- **Classification:** Maps an image to a category label.
- **Localization:** Maps an image to a bounding box and a category label.
- **Classification + Localization:** Both tasks performed simultaneously.

**Example Dataset:**
- **ImageNet Classification + Localization Challenge:** Contains 1,000 classes, each image annotated with one class and one or more bounding boxes for that class.
- **Evaluation:** At test time, the model outputs five guesses (class label + bounding box). Correct if either guess matches both the class and bounding box closely (using Intersection over Union metric).

## Localization as Regression

**Core Idea:**
- Frame bounding box prediction as a regression problem.
- **Input:** Image.
- **Output:** Four real-valued numbers (e.g., x, y coordinates of the upper-left corner and width, height of the box).
- **Loss Function:** Typically L2 (Euclidean) loss between predicted and ground truth bounding box coordinates.
- **Training:** Similar to classification networks-forward pass, compute loss, backpropagate, and update weights.

**Implementation Recipe:**
1. Use a pre-trained model (e.g., AlexNet, VGG, GoogLeNet) or train your own.
2. Remove the final fully connected (FC) layers used for classification.
3. Attach new FC layers (the "regression head") to output bounding box coordinates.
4. Train with L2 loss on bounding box coordinates.
5. At test time, use both the classification and regression heads to output class scores and bounding boxes.

**Regression Head Design Choices:**
- **Class-Agnostic Regressor:** Outputs four numbers for the bounding box, regardless of class.
- **Class-Specific Regressor:** Outputs $$C \times 4$$ numbers (one box per class). Loss is computed only for the ground truth class.
- **Attachment Point:** Could be after the last convolutional layer or after the last FC layer-both approaches are valid.

## Localizing Multiple Objects

- If you know the number of objects to localize is fixed (e.g., keypoints in human pose estimation), the regression head can output a bounding box or coordinates for each object.
- **Example:** Human pose estimation-regress (x, y) coordinates for each joint using a CNN.

**Recommendation:** If your task involves a fixed number of objects, use the localization-as-regression framework for simplicity and efficiency.

## Sliding Window Localization

**Motivation:** Simple regression-based localization works, but to achieve top performance (e.g., in competitions), more sophisticated techniques are needed.

**Sliding Window Approach:**
- Instead of running the network once per image, run it at multiple positions (windows) across the image.
- Aggregate the predictions (bounding boxes and class scores) from all windows.
- This helps correct errors and improves localization accuracy.

**OverFeat Architecture:**
- Winner of the ImageNet localization challenge (2013).
- Uses an AlexNet backbone with separate heads for classification and regression.
- Can process larger images (e.g., 257x257) by running the network on overlapping windows (e.g., four corners), then merging the resulting bounding boxes and scores.
- In practice, many more windows are used for better coverage.

## Practical Insights and Recommendations

- **Localization via regression** is simple and effective for tasks with a fixed number of objects.
- For more complex tasks (variable number of objects), sliding window and aggregation techniques (like OverFeat) are used.
- The choice between class-agnostic and class-specific regressors, and where to attach the regression head, can affect performance but are generally flexible.
- For human pose estimation and similar tasks, regression frameworks have proven effective.

---

## Key Terms & Concepts for MCQs

- **Convolutional Neural Network (CNN):** Neural network using convolutional layers for feature extraction.
- **Pooling:** Downsampling operation in CNNs.
- **Bounding Box:** Rectangle specifying the location of an object in an image.
- **Regression Head:** Network component outputting continuous values (e.g., bounding box coordinates).
- **Class-Agnostic vs. Class-Specific Regression:** Whether the bounding box prediction depends on the class label.
- **Sliding Window:** Technique of applying a model to multiple subregions of an image.
- **OverFeat:** Early architecture using sliding window localization.
- **Intersection over Union (IoU):** Metric for evaluating bounding box overlap.

---

## Study Tips

- Understand the difference between classification, localization, and detection tasks.
- Be able to explain how bounding box regression works and why it's used for localization.
- Know the main CNN architectures and their historical significance.
- Be familiar with the sliding window approach and why it's useful for detection/localization.
- Practice drawing diagrams of CNN architectures and localization pipelines to solidify understanding.

These notes should provide a comprehensive foundation for both studying the material and answering open-book MCQs on the topic.
