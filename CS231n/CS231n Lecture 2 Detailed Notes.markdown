# Lecture 2: Image Classification and Linear Classifiers

These notes cover Lecture 2, focusing on image classification, nearest neighbor classifiers, and linear classifiers. Each topic is explained in bullet points, first in **Technical Language** for advanced learners, then in **Simpler Language** for beginners.

## 1. Administrative Announcements

- **Technical Language**:
  - **Recording**: Lectures are recorded, capturing audio; students’ voices may be included, though they are not visually recorded.
  - **Display Issue**: Projector screen has aspect ratio distortion, mitigated by the human visual cortex’s spatial invariance.
  - **First Assignment**:
    - Released tonight or early tomorrow, due January 20 (two weeks).
    - Tasks: Implement a k-Nearest Neighbor (k-NN) classifier, a linear classifier, and a two-layer neural network with backpropagation.
    - Material covered over the next two weeks.
    - Warning: Assignments differ from 2015 to prevent reuse.
  - **Programming Environment**:
    - Uses Python with NumPy for efficient matrix and vector operations.
    - Assumes familiarity with Python/NumPy; a tutorial by Justin is available on the course website.
    - Requires optimized vectorized NumPy code for CPU performance.
  - **Terminal.com**:
    - Cloud-based virtual machine platform with pre-configured assignment dependencies and datasets.
    - Accessible via web interface with Jupyter notebooks and terminal.
    - Supports CPU and optional GPU instances (GPU not mandatory, e.g., for CUDA).
    - Paid service; credits provided upon emailing a TA, with responsible usage monitored.

- **Simpler Language**:
  - **Recording**: The lecture is recorded, so your voice might be in it if you talk, but you won’t be on camera.
  - **Screen Problem**: The screen looks stretched, but our brains can handle it, so it’s okay.
  - **First Assignment**:
    - Out tonight or tomorrow, due in two weeks (January 20).
    - You’ll code: a k-NN classifier (finds similar pictures), a linear classifier (uses math to decide), and a small neural network (learns smartly).
    - We’ll learn this stuff in the next two weeks.
    - Don’t use last year’s assignments—they’re different.
  - **Coding Setup**:
    - You’ll use Python and NumPy to do fast math with pictures.
    - You need to know some Python/NumPy; if not, check Justin’s tutorial on the website.
    - Your code should be fast for regular computers.
  - **Terminal.com**:
    - A website to run your code on cloud computers, with everything set up.
    - Use it through your browser; has fast (GPU) options, but you don’t need them.
    - Costs money, but we’ll give you free credits—email a TA to get them, and don’t waste them (we’re tracking).

## 2. Introduction to Image Classification

- **Technical Language**:
  - **Definition**: Assigns a categorical label (e.g., dog, cat, truck) to an image, represented as a pixel intensity array (e.g., $300 \times 100 \times 3$ for RGB).
  - **Significance**: Foundational for computer vision tasks like object detection, image captioning, and segmentation.
  - **Semantic Gap**: Challenge of mapping low-level pixels to high-level concepts (e.g., "cat").
  - **Challenges**:
    - **Viewpoint Variations**: Camera rotation, zoom, or translation alters pixel patterns, requiring invariance.
    - **Illumination**: Lighting changes modify pixel intensities.
    - **Deformation**: Objects (e.g., cats) vary in pose.
    - **Occlusion**: Partial visibility (e.g., cat behind curtain) obscures features.
    - **Background Clutter**: Complex backgrounds obscure objects.
    - **Intraclass Variation**: Diverse appearances within a class (e.g., cat breeds).
  - **Complexity**: Combinatorial variations make robust classification difficult, yet modern systems achieve near-human accuracy in milliseconds.

- **Simpler Language**:
  - **What It Is**: Figuring out what a picture shows (e.g., dog, cat, truck) by looking at its pixels (numbers).
  - **Why It Matters**: Key for teaching computers to see, helping with things like finding objects or describing pictures.
  - **Big Problem**: Pixels are just numbers, but we need to understand them as things like “cat.”
  - **Why It’s Hard**:
    - **Camera Angles**: Pictures change if the camera moves or zooms.
    - **Lighting**: Bright or dim light changes how pictures look.
    - **Poses**: Cats can be in all sorts of positions.
    - **Hidden Parts**: A cat might be partly covered (e.g., behind something).
    - **Messy Backgrounds**: Stuff in the background can hide the main thing.
    - **Different Looks**: Cats come in many types (e.g., different breeds).
  - **Amazing Fact**: Even with all these issues, computers can guess what’s in a picture super fast, almost as good as people.

## 3. Early Approaches to Image Classification

- **Technical Language**:
  - **Explicit Methods**: Early vision used hand-crafted rules (e.g., detect cat ears via edge tracing, shape classification, or texture analysis).
  - **Limitation**: Unscalable; new classes (e.g., boats) required redesigned rules due to varying features.
  - **Data-Driven Approach**: Modern machine learning leverages large datasets (e.g., internet images labeled via text).
  - **Process**:
    - Training phase: Provide labeled examples.
    - Model training: Learn patterns.
    - Testing phase: Classify unseen images.
  - **Enabler**: Internet-scale data, unlike early systems with low-resolution, limited images.

- **Simpler Language**:
  - **Old Way**: People wrote specific rules for computers, like “look for cat ears” or “check shapes.”
  - **Problem**: Didn’t work well—every new thing (like boats) needed new rules, which was too much work.
  - **New Way**: Computers learn from lots of pictures (e.g., cat pictures from the internet).
  - **How It Works**:
    - Show the computer labeled pictures.
    - It learns what they are.
    - It guesses what new pictures show.
  - **Why Better**: We have tons of pictures now, unlike before when we had only a few blurry ones.

## 4. Nearest Neighbor Classifier

- **Technical Language**:
  - **Concept**: Non-parametric method assigning a test image’s label by finding the most similar training image.
  - **Dataset**: CIFAR-10 (10 classes, 50,000 training images, 10,000 test images, 32x32 pixels).
  - **Algorithm**:
    - Training: Store all images and labels.
    - Testing: Compute similarity to all training images, transfer nearest label.
  - **Distance Metrics**:
    - Manhattan (L1): $$\sum |I_1[i] - I_2[i]|$$.
    - Euclidean (L2): $$\sqrt{\sum (I_1[i] - I_2[i])^2}$$.
  - **Implementation**: Python/NumPy with vectorized operations for efficiency.
  - **Performance**:
    - Training: Instantaneous (stores data).
    - Testing: Linearly slower with larger training sets.
  - **Speed-Up**: Approximate methods (e.g., FLANN) improve efficiency.

- **Simpler Language**:
  - **What It Does**: Looks at a new picture, finds the most similar picture it knows, and gives it the same name.
  - **Dataset**: CIFAR-10, with 10 types of things (e.g., cats, cars) and lots of small pictures (50,000 to learn, 10,000 to test).
  - **How It Works**:
    - Save all pictures and their names.
    - For a new picture, check which saved picture is most similar and use its name.
  - **Measuring Similarity**:
    - Manhattan: Add up pixel differences.
    - Euclidean: A fancier math way to compare pixels.
  - **Coding**: Uses Python to make it fast.
  - **Speed**:
    - Learning is quick (just saves pictures).
    - Checking new pictures slows down if you have more saved pictures.
  - **Trick**: Tools like FLANN make it faster.

### k-Nearest Neighbor (k-NN) Classifier

- **Technical Language**:
  - **Generalization**: Retrieves $k$ nearest training images, assigns majority label (e.g., $k=5$).
  - **Effect**: Higher $k$ smooths decision boundaries, reducing outlier impact, visualized in 2D decision regions.
  - **Hyperparameters**:
    - Distance metric (L1, L2).
    - $k$ (e.g., 1, 3, 5, 10).
  - **Tuning**: Dataset-dependent; requires cross-validation.
  - **Cross-Validation**:
    - Five-fold: Split training data into five folds, train on four, validate on one, rotate, average performance.
    - Selects optimal hyperparameters (e.g., $k=7$).
  - **Testing**: Evaluate once on test set to avoid overfitting.

- **Simpler Language**:
  - **What’s Different**: Instead of one similar picture, it picks the top $k$ (e.g., 5) and lets them vote on the name.
  - **Why It Helps**: More votes make it less likely to pick a weird, wrong picture.
  - **Choices to Make**:
    - How to measure “similar” (e.g., Manhattan or Euclidean).
    - How many pictures to check ($k$, like 1, 3, or 5).
  - **Picking Choices**: Depends on the pictures; we test options to find the best.
  - **Testing Fairly**:
    - Split learning pictures into five groups, try options on each, and pick what works best.
    - Example: Might find $k=7$ is best.
  - **Final Check**: Use test pictures only once to get the real score, so we don’t cheat.

### Limitations of k-NN

- **Technical Language**:
  - **Inefficiency**: Test time scales linearly with training set size due to exhaustive comparisons.
  - **Distance Metrics**: L1/L2 are unintuitive in high-dimensional spaces; transformations (e.g., shifts, darkening) yield similar distances despite perceptual differences.
  - **Practicality**: Rarely used due to inefficiency and semantic limitations, but educational for data-driven concepts.

- **Simpler Language**:
  - **Too Slow**: Takes longer to check new pictures if you have lots of saved ones.
  - **Bad Similarity**: The way it compares pixels doesn’t always make sense—moving or dimming a picture can trick it.
  - **Not Used Much**: Not great for big projects, but good for learning how computers use pictures.

## 5. Linear Classification

- **Technical Language**:
  - **Concept**: Parametric method mapping images to class scores via a learnable function, $f(X, W, b) = W \cdot X + b$.
  - **Dataset**: CIFAR-10; image flattened to $X$ (32x32x3 = 3072 dimensions).
  - **Function**:
    - $W$: 10x3072 weight matrix, each row a class template.
    - $b$: 10-dimensional bias vector.
    - Outputs 10 class scores.
  - **Parameters**: 30,720 ($W$) + 10 ($b$) = 30,730.
  - **Interpretations**:
    - Template matching: Dot product with $W$ rows.
    - Weighted pixel sums.
    - Hyperplane separation in 3072D space.
    - Projection to label space.
  - **Handling Variations**:
    - Resize images to fixed size (e.g., 32x32).
    - Serialize RGB channels consistently.
    - Data augmentation (e.g., jittering) for robustness.
  - **Significance**: Foundation for neural networks and CNNs.

- **Simpler Language**:
  - **What It Does**: Uses a math formula to give each possible answer (e.g., cat, car) a score, picking the highest.
  - **Dataset**: CIFAR-10; turns pictures into a list of 3072 numbers (pixels).
  - **Formula**: $f(X, W, b) = W \cdot X + b$
    - $W$: A big table of numbers, like a recipe for each thing (e.g., blue for planes).
    - $b$: A small tweak to adjust scores.
    - Gives 10 scores, one for each type.
  - **Numbers to Learn**: About 30,730 numbers in $W$ and $b$.
  - **How It Thinks**:
    - Checks if a picture matches a recipe (e.g., lots of blue).
    - Adds up pixel numbers with importance.
    - Draws invisible lines to separate things in a math world.
    - Turns picture numbers into answer scores.
  - **Dealing with Differences**:
    - Make all pictures the same size.
    - Arrange pixel colors the same way.
    - Tweak pictures (e.g., rotate) to learn better.
  - **Why Important**: First step to smarter systems like neural networks.

### Limitations of Linear Classifiers

- **Technical Language**:
  - **Mode Limitation**: Cannot model multiple class modes (e.g., red vs. yellow cars), yielding averaged templates (e.g., “two-headed” horse).
  - **Sensitivity**:
    - Non-linear transformations (e.g., warping).
    - Grayscale images (lose color cues).
    - Negative images (inverted intensities).
    - Spatially invariant patterns.
  - **Challenging Test Sets**:
    - Similar centroids (e.g., scooters vs. motorcycles).
    - Non-linearly separable distributions.
  - **Comparison**: Optimizes $W$ for a loss function, outperforming simple averaging of class images.

- **Simpler Language**:
  - **Can’t Handle Variety**: Struggles with different looks (e.g., red cars but not yellow), making mixed-up recipes (e.g., a horse with two heads).
  - **Gets Confused By**:
    - Twisted or rotated pictures.
    - Pictures without color (grayscale).
    - Reversed-color pictures.
    - Things that move around in the picture.
  - **Tough Tests**:
    - Things that look too similar (e.g., scooters and motorcycles).
    - Weirdly arranged pictures.
  - **Better Than Averaging**: It’s smarter than just averaging pictures, as it learns the best recipe.

## 6. Upcoming Topics

- **Technical Language**:
  - **Loss Functions**: Quantify performance of $W$ on training data; low loss indicates correct classification.
  - **Optimization**: Iteratively minimize loss using gradient-based methods, starting from random $W$.
  - **Neural Networks**: Stack layers to model complex patterns (e.g., multiple car colors).
  - **Convolutional Neural Networks (CNNs)**: Enhance spatial structure handling, maintaining the score-loss-optimization framework.

- **Simpler Language**:
  - **Loss Functions**: A score to tell how good or bad our recipe ($W$) is; low score means we’re doing great.
  - **Optimization**: Slowly tweak the recipe using math to make it better, starting with a random guess.
  - **Neural Networks**: Add more layers to learn tricky things (e.g., all car colors).
  - **Convolutional Neural Networks**: Make it even better at understanding pictures, using the same idea of scores and learning.

## 7. Questions and Discussions

- **Technical Language**:
  - **k-NN Training Accuracy**:
    - 1-NN (L1/L2): 100%, as each training image matches itself.
    - k-NN (e.g., $k=5$): Not 100%, as majority voting may override correct labels.
  - **Unbalanced Datasets**: Biases ($b$) may favor overrepresented classes, depending on loss function.
  - **Hyperparameter Tuning**: Test set tuning causes overfitting; use cross-validation for generalization.
  - **Iterative Optimization**: Direct solutions for optimal $W$ are intractable; gradient descent incrementally improves weights.

- **Simpler Language**:
  - **k-NN on Its Own Pictures**:
    - One neighbor: Always right, because it finds itself.
    - Five neighbors: Might be wrong if nearby pictures vote differently.
  - **Uneven Data**: If we have more cats, the system might lean toward cats, but it depends on how we score it.
  - **Choosing Options**: Don’t test on final pictures—it’s cheating; test on practice pictures to be fair.
  - **Why Slow Learning**: We can’t find the perfect recipe instantly, so we tweak it step by step with math.

## 8. Key Takeaways

- **Technical Language**:
  - **Image Classification**: Core task with challenges like viewpoint and illumination variations.
  - **k-NN**: Non-parametric, inefficient, limited by high-dimensional distance metrics.
  - **Linear Classifier**: Parametric, maps images to scores via $W$, interpreted as templates or hyperplanes.
  - **Hyperparameter Tuning**: Cross-validation ensures generalization.
  - **Limitations**: Linear classifiers fail on complex variations, necessitating neural networks.
  - **Future**: Course will cover loss functions, optimization, neural networks, and CNNs.

- **Simpler Language**:
  - **Image Classification**: Teaching computers to name pictures, despite tricky changes like lighting.
  - **k-NN**: Finds similar pictures but is slow and gets confused by pixel math.
  - **Linear Classifier**: Uses a math recipe to score answers, like checking for blue skies.
  - **Picking Options**: Test fairly to make sure it works on new pictures.
  - **Problems**: Can’t handle all picture types, so we need smarter systems.
  - **Next Steps**: Learn how to score recipes, improve them, and build better systems like neural networks.
