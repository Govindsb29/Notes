# Lecture 2: Image Classification and Linear Classifiers

## 1. Administrative Announcements
- **Recording Notice**: Lectures are recorded. Students uncomfortable with their voices being recorded should note that while they are not in the camera's view, their voices may be captured.
- **Screen Issue**: The projector screen is stretched, but this is not a significant issue due to the human visual cortex's adaptability to distortions.
- **Assignments**:
  - **First Assignment**: Released tonight or early tomorrow, due January 20 (two weeks).
    - **Tasks**: Implement a k-Nearest Neighbor (k-NN) classifier, a linear classifier, and a two-layer neural network, including the backpropagation algorithm for the neural network.
    - **Coverage**: Material will be covered in the next two weeks.
    - **Warning**: Assignments differ from last year (2015). Do not use old assignments.
  - **Programming Environment**:
    - **Language**: Python with NumPy for optimized matrix and vector operations.
    - **Requirement**: Assumes familiarity with Python and NumPy. A tutorial by Justin is available on the course website for those unfamiliar with NumPy expressions.
    - **Example**: Students should understand optimized NumPy code for matrix manipulations to ensure CPU efficiency.
  - **Terminal.com**:
    - A cloud-based virtual machine service for running assignments.
    - Pre-installed with all dependencies and datasets.
    - Accessible via a web browser with a Jupyter notebook and terminal interface.
    - Offers CPU and GPU machines (GPU usage optional, e.g., for CUDA code, but not required).
    - Normally paid, but credits will be distributed. Contact a TA to request credits, and use them responsibly as usage is tracked.

## 2. Introduction to Image Classification
- **Definition**: Assign a label from a fixed set of categories (e.g., dog, cat, truck, plane) to an input image, represented as a grid of pixel values (numbers).
- **Significance**:
  - A foundational task in computer vision. Understanding image classification enables tackling related tasks like object detection, image captioning, and segmentation.
  - Provides conceptual grounding for broader computer vision applications.

### Challenges in Image Classification
- **Semantic Gap**: Images are large arrays of numbers (e.g., 300x100x3 for RGB), but the task requires mapping these to high-level concepts like "cat." This gap between raw pixels and semantic meaning is a core difficulty.
- **Specific Challenges**:
  - **Viewpoint Variations**: Camera rotations, zooming, or shifts change pixel patterns, requiring robustness.
  - **Illumination**: Lighting changes (e.g., a cat in bright vs. dim light) alter pixel brightness.
  - **Deformation**: Objects like cats appear in various poses, requiring recognition despite shape changes.
  - **Occlusion**: Partial visibility (e.g., a cat behind a curtain) complicates recognition.
  - **Background Clutter**: Objects may blend into complex backgrounds.
  - **Intraclass Variation**: Variations within a class (e.g., different cat breeds) add complexity.
- **Complexity**: The combination of these challenges creates a vast range of image variations, making it remarkable that modern classifiers achieve near-human accuracy in milliseconds.

## 3. Early Approaches to Image Classification
- **Explicit Approaches**:
  - Early computer vision used hand-crafted rules, e.g., detecting cat ears by tracing edges, classifying shapes, or analyzing textures.
  - **Problem**: Unscalable. Each new class (e.g., boat) requires redesigning rules, as features like edge arrangements differ.
- **Data-Driven Approach**:
  - Modern machine learning approach leveraging large datasets (e.g., internet images labeled via surrounding text).
  - **Process**:
    1. **Training Phase**: Provide labeled training examples (e.g., images of cats labeled as "cat").
    2. **Model Training**: Train a model to recognize patterns.
    3. **Testing Phase**: Use the model to classify new, unseen images.
  - Enabled by large datasets, unlike early systems with limited, low-resolution images.

## 4. Nearest Neighbor Classifier
- **Concept**: A non-parametric classifier that compares a test image to all training images and assigns the label of the most similar (nearest) training image.
- **Dataset Example**: CIFAR-10
  - 10 classes (e.g., airplane, car, cat).
  - 50,000 training images, 10,000 test images.
  - Images are small (32x32 pixels), a toy dataset.
- **Algorithm**:
  - **Training**: Store all training images and labels.
  - **Testing**: Compute similarity to all training images and transfer the label of the nearest one.
- **Distance Metric**:
  - **Manhattan (L1) Distance**: Sum of absolute differences between pixel values.
    - Formula: For images \( I_1 \), \( I_2 \), distance = \( \sum |I_1[i] - I_2[i]| \).
    - Zero distance for identical images.
  - **Euclidean (L2) Distance**: Sum of squared differences, emphasizing larger deviations.
- **Implementation**:
  - Python/NumPy code using vectorized operations for efficiency.
    - **Training**: Store images (\( X \)) and labels (\( Y \)).
    - **Testing**: Compute distances to all training images in one line, find the minimum distance, and predict the corresponding label.
- **Performance**:
  - **Training Time**: Instantaneous (stores data).
  - **Test Time**: Linearly slower with larger training sets, as each test image requires comparison to all training images.
  - **Issue**: Inefficient for large datasets, unlike neural networks with constant test-time complexity.
- **Speed-Up**: Approximate nearest neighbor methods (e.g., FLANN library) improve efficiency.

### k-Nearest Neighbor (k-NN) Classifier
- **Generalization**: Retrieve the \( k \) nearest neighbors and perform a majority vote on their labels.
- **Example**:
  - For \( k=5 \), retrieve five most similar images and assign the most common label.
  - Visualized with a 2D dataset, showing decision regions (areas assigned to each class).
- **Effect of \( k \)**:
  - Higher \( k \) smooths decision boundaries, reducing outlier impact.
  - Improves performance but introduces a hyperparameter (\( k \)).
- **Hyperparameters**:
  - **Distance Metric**: L1, L2, or others.
  - **\( k \)**: Number of neighbors (e.g., 1, 3, 5, 10).
  - **Problem**: Optimal values are dataset-dependent, requiring tuning.

### Hyperparameter Tuning and Cross-Validation
- **Incorrect Approach**: Tuning on the test set leads to overfitting, as the test set evaluates generalization.
- **Correct Approach**: Use training set for tuning via **cross-validation**:
  - **Five-Fold Cross-Validation**:
    - Split training data into five folds.
    - Train on four folds, validate on the fifth.
    - Rotate validation fold and repeat.
    - Average performance to select best hyperparameters (e.g., \( k=7 \)).
  - **Validation Set**: Alternatively, use a single held-out validation set.
- **Process**:
  - Try various hyperparameters (e.g., \( k \), distance metric).
  - Evaluate on validation set/folds.
  - Select best hyperparameters.
  - Evaluate once on test set for final accuracy.
- **Why Cross-Validation?**:
  - Ensures hyperparameters generalize to unseen data.
  - Prevents overfitting to the test set.

### Limitations of k-NN
- **Inefficiency**: Slow at test time due to linear scaling.
- **Distance Metrics**: Pixel-based distances (L1, L2) are unintuitive in high-dimensional spaces.
  - Example: Shifted, darkened, or partially altered images may have the same L2 distance, despite perceptual differences.
  - High-dimensional spaces cause distance metrics to lose semantic meaning.
- **Practical Use**: Rarely used due to inefficiencies and limitations, but useful for understanding data-driven approaches and train/test splits.

## 5. Linear Classification
- **Transition**: Moving to parametric approaches, building toward convolutional neural networks (CNNs).
- **Motivation**:
  - **Task-Based**: Focus on computer vision tasks.
  - **Model-Based**: Introduce deep learning and neural networks, applicable to vision, speech, translation, control (e.g., Atari games).
  - Neural networks are modular, like "Lego blocks," enabling flexible component stacking.
- **Example Application**: Image captioning (lecturer’s prior work).
  - **Task**: Generate a sentence describing an image (e.g., "A man in a black shirt is playing a guitar").
  - **Model**: Combines a CNN (vision) with a recurrent neural network (RNN, sequence modeling).
  - **Process**: CNN processes image, RNN generates sentence, gradients optimize both.

### Parametric vs. Non-Parametric Approaches
- **Non-Parametric (k-NN)**: No parameters to optimize; relies on storing and comparing data.
- **Parametric (Linear Classifier)**: Defines a function with learnable parameters (weights) to map images to class scores.

### Linear Classifier
- **Goal**: Map an image to scores for each class, with the correct class having the highest score.
- **Function**:
  - **Input**: Image \( X \), flattened into a column vector (e.g., 32x32x3 = 3072 dimensions for CIFAR-10).
  - **Output**: Scores for each class (e.g., 10 for CIFAR-10).
  - **Function**: \( f(X, W, b) = W \cdot X + b \)
    - \( W \): Weight matrix (10x3072), each row for a class.
    - \( b \): Bias vector (10 dimensions), adjusting baseline scores.
    - \( W \cdot X \): Matrix multiplication, computing weighted sums of pixels.
- **Parameters**:
  - \( W \): 30,720 parameters (10x3072).
  - \( b \): 10 parameters.
  - **Total**: 30,730 learnable parameters.
- **Interpretation**:
  - **Template Matching**: Each row of \( W \) is a class template, computing a dot product to measure similarity.
    - Example: Plane template has positive weights in blue channel, detecting blue regions.
    - Visualized by reshaping \( W \) rows into images, showing class-specific patterns.
  - **Weighted Sum**: Scores are weighted sums of pixel values, with weights determining pixel importance.
  - **Spatial Interpretation**: \( W \) defines hyperplanes in high-dimensional pixel space, separating classes.
    - Zero-score hyperplane: Where a class score is zero.
    - Gradient direction: Indicates increasing "classness" (e.g., "carness").
  - **Projection**: Maps image space to label space.

### Handling Image Variations
- **Color Channels**: Images serialized consistently (e.g., stacking red, green, blue channels).
- **Different Sizes**: Resize images to a fixed size (e.g., 32x32). Square images are standard; resizing non-square images may degrade performance.
- **Data Augmentation**: Augment training data with transformed versions (e.g., jittering, stretching) to handle variations.

### Limitations of Linear Classifiers
- **Inability to Capture Modes**:
  - Example: Car template may emphasize red due to dataset bias, missing yellow cars.
  - Horse template may combine left- and right-facing horses, creating a "two-headed" template.
  - Lacks capacity to model multiple modes (e.g., colors, orientations).
- **Sensitivity to Transformations**:
  - **Grayscale Images**: Poor performance without color, as they rely on color patterns.
  - **Non-Linear Variations**: Fail on warped or rotated images.
  - **Negative Images**: Inverted colors score poorly despite retaining shape.
  - **Spatial Invariance**: Struggle with objects in different positions.
- **Hard Test Sets**:
  - Classes with similar centroids (e.g., scooters vs. motorcycles).
  - Non-linearly separable patterns (e.g., concentric blobs).
  - Spatially invariant textures or patterns.

### Comparison to Averaging
- **Averaging Approach**: Mean pixel values of class images as a template.
- **Difference**: Linear classifiers optimize \( W \) to minimize a loss function, not just average images. Averaging is a heuristic but less optimal.

## 6. Upcoming Topics
- **Loss Functions**:
  - Quantify \( W \) performance on training set.
  - Low loss indicates correct classification; high loss indicates errors.
  - Example: Penalize when correct class score is lower than others.
- **Optimization**:
  - Iteratively adjust \( W \) to minimize loss using gradient-based methods.
  - Start with random \( W \), compute gradients, update weights.
  - Unlike k-NN, computationally intensive at training but efficient at test time.
- **Neural Networks**:
  - Stack multiple layers to detect complex patterns (e.g., different car colors).
  - Layers learn specific features, combined later for robust classification.
- **Convolutional Neural Networks (CNNs)**:
  - Handle spatial structures, improving image classification.
  - Same framework: map images to scores, compute loss, optimize parameters.

## 7. Questions and Discussions
- **Accuracy of k-NN on Training Data**:
  - **1-NN with L1/L2 Distance**: 100% accuracy, as each training image matches itself.
  - **k-NN (e.g., \( k=5 \))**: Not 100%, as majority voting may override correct labels.
- **Unbalanced Datasets**:
  - Biases (\( b \)) may be higher for overrepresented classes, but impact depends on loss function.
- **Hyperparameters and Cheating**:
  - Tuning on test set is invalid, as it overfits.
  - Hyperparameters are dataset-specific, tested via cross-validation.
- **Why Iterative Optimization?**:
  - Direct solutions for optimal \( W \) are intractable for complex models.
  - Gradient descent incrementally improves weights, guided by loss gradients.

## 8. Key Takeaways
- **Image Classification**: Core task with challenges like viewpoint, illumination, and deformation.
- **k-NN Classifier**: Simple but inefficient, limited by unintuitive distance metrics.
- **Linear Classifier**: Parametric, mapping images to scores via weights, interpreted as templates or hyperplanes.
- **Hyperparameter Tuning**: Use cross-validation for generalization.
- **Limitations**: Linear classifiers struggle with complex variations, necessitating neural networks.
- **Future Direction**: Course will cover neural networks and CNNs, using loss functions and optimization.