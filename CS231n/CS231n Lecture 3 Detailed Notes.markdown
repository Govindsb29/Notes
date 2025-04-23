# Lecture 3: Loss Functions and Optimization

These notes cover Lecture 3, focusing on loss functions (SVM and softmax), regularization, and optimization for image classification. Each topic is explained in bullet points, first in **Technical Language** for advanced learners, then in **Simpler Language** for beginners.

## 1. Administrative Announcements

- **Technical Language**:
  - **Assignment Deadline**: First assignment due next Wednesday (nine days from lecture date). Monday is a holiday, with no class or office hours, requiring careful time planning.
  - **Late Days**: Students can allocate late days across assignments as needed.

- **Simpler Language**:
  - **Assignment Due**: You have nine days until next Wednesday to finish the first assignment. No class or help sessions on Monday (it’s a holiday), so plan your time well.
  - **Extra Time**: You can use late days if you need more time for assignments.

## 2. Recap of Image Classification

- **Technical Language**:
  - **Problem Context**: Image classification is a complex visual recognition task, requiring robustness to variations (e.g., viewpoint, illumination, deformation, occlusion, background clutter, intraclass variation).
  - **Challenge**: The combinatorial cross-product of these variations creates an intractable problem space.
  - **Achievements**: State-of-the-art methods achieve near-human or superhuman accuracy for thousands of categories, running in near real-time on mobile devices, with significant progress in the last three years.
  - **Data-Driven Approach**: Explicit rule-based classifiers are infeasible; training from data with train/validation/test splits is standard.
  - **Previous Topics**:
    - Nearest Neighbor and k-NN classifiers on CIFAR-10 dataset (10 classes, 50,000 training images, 10,000 test images, 32x32 pixels).
    - Parametric approach: Linear classifier $f(X, W) = W \cdot X$, mapping images to class scores, interpreted as template matching or hyperplane separation.
  - **Current Focus**: Define a loss function to quantify classifier performance and optimize weights $W$ to minimize loss.

- **Simpler Language**:
  - **What’s Hard**: Figuring out what’s in a picture (like a cat) is tough because pictures change a lot (different angles, lighting, poses, hidden parts, messy backgrounds, or different cat types).
  - **Why It’s Tricky**: There are so many ways pictures can look, it feels impossible to handle them all.
  - **Cool Fact**: Today’s tech can name things in pictures almost as well as humans, even for thousands of things, and it works super fast on your phone—all in just the last three years!
  - **How We Do It**: Instead of writing rules, we teach computers using lots of pictures, splitting them into groups for learning, testing, and fine-tuning.
  - **What We Learned Before**:
    - Nearest Neighbor and k-NN: Find similar pictures to guess what’s in a new one, using CIFAR-10 (10 types, lots of small pictures).
    - Linear Classifier: Uses a math formula ($f(X, W) = W \cdot X$) to give scores for each type, like matching patterns or drawing lines to separate things.
  - **Today’s Goal**: Figure out how to score how good or bad our guesses are and make the computer better at guessing.

## 3. Loss Functions: Multi-Class SVM Loss

- **Technical Language**:
  - **Purpose**: Quantifies the “unhappiness” of a classifier’s scores, enabling optimization to find optimal weights $W$.
  - **Setup**: For a training example $(X_i, y_i)$, scores are $s = W \cdot X_i$, where $s_{y_i}$ is the correct class score, and $s_j$ are incorrect class scores.
  - **SVM Loss Formula**: For example $i$, loss is:
    $$L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + 1)$$
    Total loss: $$L = \frac{1}{N} \sum_{i=1}^N L_i$$
  - **Interpretation**:
    - Compares incorrect scores $s_j$ to correct score $s_{y_i}$, enforcing a margin of 1.
    - If $s_j < s_{y_i} - 1$, no loss (margin satisfied); otherwise, loss is the violation amount.
    - Margin of 1 is scale-invariant due to $W$’s arbitrary scaling; other margins alter the solution non-trivially.
  - **Example**:
    - Scores: Cat (correct, 3.2), Car (5.1), Frog (-1.7).
    - Loss: $\max(0, 5.1 - 3.2 + 1) + \max(0, -1.7 - 3.2 + 1) = 2.9 + 0 = 2.9$.
    - Intuitively, Car’s score (5.1) exceeds desired threshold ($3.2 + 1 = 4.2$), contributing 2.9 to loss; Frog’s score is low, contributing 0.
  - **Properties**:
    - Minimum loss: 0 (all margins satisfied).
    - Maximum loss: Infinite (correct score arbitrarily low, incorrect scores high).
    - Initial loss (small random $W$, scores near 0): Approximately $C - 1$ (for $C$ classes), e.g., 2 for 3 classes.
  - **Implementation**: Vectorized NumPy code computes margins ($s_j - s_{y_i} + 1$), sets correct class margin to 0, and sums non-negative values.
  - **Variants**:
    - Summing over all classes (including correct): Adds constant 1 to loss, no impact on optimization.
    - Mean instead of sum over incorrect classes: Scales loss by $1/(C-1)$, no impact on optimal $W$.
    - Squared hinge loss: $\sum_{j \neq y_i} \max(0, s_j - s_{y_i} + 1)^2$. Alters trade-offs non-linearly, yielding different optimal $W$; a hyperparameter choice.

- **Simpler Language**:
  - **Why We Need It**: We need a way to say how bad our guesses are so we can make them better.
  - **How It Works**: For each picture, we give scores to each possible answer (e.g., cat, car, frog). The right answer should score higher than wrong ones by at least 1.
  - **Math Rule**: For a picture, add up how much wrong answers beat the right answer plus 1, but only count it if it’s more than 0. Average this over all pictures.
    - Looks like: $$L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + 1)$$
  - **What It Means**:
    - If a wrong answer’s score is too high, it adds to our “badness” score.
    - The “1” is like a safety gap to make sure the right answer is clearly better.
    - We can stretch or shrink our math recipe ($W$), so 1 is just a handy choice.
  - **Example**:
    - Scores: Cat (right, 3.2), Car (5.1), Frog (-1.7).
    - Cat should beat Car by 1 (want Car < 4.2), but Car’s 5.1, so badness is 2.9. Frog’s low, so no badness. Total: 2.9.
  - **Key Facts**:
    - Best score: 0 (all right answers win by enough).
    - Worst score: Infinite (if right answer scores super low and wrong ones high).
    - Starting score (random recipe): About 2 for 3 types, good for checking if our code’s right.
  - **Coding**: We use Python to quickly calculate these scores and badness, ignoring the right answer’s own score to avoid mistakes.
  - **Other Ways**:
    - Counting the right answer: Just adds 1 to badness, doesn’t change the best recipe.
    - Averaging instead of adding wrong answers: Makes badness smaller, but doesn’t change the best recipe.
    - Squaring the badness: Changes how we balance things, giving a different recipe; we pick which to use.

## 4. Regularization

- **Technical Language**:
  - **Issue with SVM Loss**: Zero-loss $W$ is not unique; scaling $W$ by $\alpha > 1$ increases score margins but yields identical loss, creating an undesirable subspace of equivalent solutions.
  - **Solution**: Add a regularization term to the loss: $$L = \frac{1}{N} \sum_{i=1}^N L_i + \lambda R(W)$$
  - **Purpose**: Regularization imposes a preference for certain $W$, balancing data fit (data loss) and intrinsic properties of $W$ (regularization loss), improving test set generalization.
  - **L2 Regularization**:
    - Form: $R(W) = \sum_k \sum_l W_{k,l}^2$ (sum of squared weights).
    - Favors small, diffuse weights, minimizing $W$ magnitude.
  - **Example**:
    - Input $X = [1, 1, 1, 1]$, two weights: $W_1 = [1, 0, 0, 0]$, $W_2 = [0.25, 0.25, 0.25, 0.25]$.
    - Both yield same score ($W \cdot X = 1$), but $R(W_1) = 1$, $R(W_2) = 0.25$, favoring $W_2$.
  - **Intuition**:
    - Diffuse weights consider more input features, enhancing robustness.
    - Encourages generalization by preventing over-reliance on specific features.
  - **Trade-Off**: May increase training error but improves test error, as diffuse weights generalize better.
  - **Other Forms**: L1 regularization induces sparsity (many weights = 0), used for feature selection; covered later.

- **Simpler Language**:
  - **Problem**: Our math recipe ($W$) can be stretched to make scores bigger, but it doesn’t change how bad our guesses are, which isn’t great.
  - **Fix**: Add an extra rule to our badness score: $$L = \text{average badness} + \lambda \times \text{recipe niceness}$$
  - **Why It Helps**: This extra rule makes us pick a recipe that’s not just good at guessing training pictures but also “nice,” so it works better on new pictures.
  - **L2 Rule**:
    - Niceness = add up all recipe numbers squared.
    - Likes small, spread-out numbers in the recipe.
  - **Example**:
    - Picture numbers: [1, 1, 1, 1]. Two recipes: $W_1 = [1, 0, 0, 0]$, $W_2 = [0.25, 0.25, 0.25, 0.25]$.
    - Both give same score (1), but $W_2$ is “nicer” because its numbers are smaller and spread out.
  - **Why Spread-Out Is Good**:
    - Uses more parts of the picture, not just one, making guesses more reliable.
    - Helps with new pictures we haven’t seen.
  - **Balance**: Might make training guesses worse but makes new picture guesses better.
  - **Another Way**: L1 rule makes lots of recipe numbers 0, like picking only important parts; we’ll talk about it later.

## 5. Loss Functions: Softmax Classifier

- **Technical Language**:
  - **Overview**: Alternative to SVM, interprets scores as unnormalized log probabilities, generalizing multinomial logistic regression.
  - **Setup**: Scores $s = W \cdot X_i$. Softmax function converts scores to probabilities:
    $$P(y_k | X_i) = \frac{e^{s_k}}{\sum_j e^{s_j}}$$
  - **Loss**: Negative log likelihood of correct class:
    $$L_i = -\log P(y_i | X_i) = -\log \left( \frac{e^{s_{y_i}}}{\sum_j e^{s_j}} \right)$$
    Total loss: $$L = \frac{1}{N} \sum_{i=1}^N L_i + \lambda R(W)$$
  - **Example**:
    - Scores: Cat (correct, 3.2), Car (5.1), Frog (-1.7).
    - Exponentiate: $e^{3.2} \approx 24.5$, $e^{5.1} \approx 164$, $e^{-1.7} \approx 0.18$.
    - Sum: $24.5 + 164 + 0.18 \approx 188.68$.
    - Probabilities: Cat = $24.5 / 188.68 \approx 0.13$, Car = $0.87$, Frog = $0.001$.
    - Loss: $-\log(0.13) \approx 0.89$.
  - **Properties**:
    - Minimum loss: 0 (correct class probability = 1, $-\log(1) = 0$).
    - Maximum loss: Infinite (correct class probability $\to 0$, $-\log(0) \to \infty$).
    - Initial loss (small random $W$, scores near 0): $-\log(1/C)$, e.g., $\log(3) \approx 1.1$ for 3 classes.
  - **Comparison to SVM**:
    - SVM: Enforces margin of 1; loss is zero if margins are met, invariant to further score increases.
    - Softmax: Always prefers higher correct class scores (increases probability), sensitive to score differences even beyond margins.
    - SVM is robust locally; softmax considers the entire data distribution.
    - In practice, both yield similar results; softmax is often preferred (reason unclear).

- **Simpler Language**:
  - **What It Is**: Another way to score how bad our guesses are, treating scores like chances of each answer being right.
  - **How It Works**: Turn scores into probabilities (chances) using a math trick called softmax:
    $$P(\text{type} | \text{picture}) = \frac{e^{\text{score for type}}}{\text{sum of } e^{\text{all scores}}}$$
  - **Badness Score**: Take the chance of the right answer, make it a log, and make it negative:
    $$L_i = -\log(\text{chance of right answer})$$
    Average this and add the niceness rule.
  - **Example**:
    - Scores: Cat (right, 3.2), Car (5.1), Frog (-1.7).
    - Use $e$ (math number) to get big numbers: $e^{3.2} \approx 24.5$, $e^{5.1} \approx 164$, $e^{-1.7} \approx 0.18$.
    - Add them: $188.68$. Chances: Cat = $24.5 / 188.68 \approx 13\%$, Car = $87\%$, Frog = $0.1\%$.
    - Badness: $-\log(0.13) \approx 0.89$ (low chance for cat = bad).
  - **Key Facts**:
    - Best score: 0 (right answer gets 100% chance).
    - Worst score: Infinite (right answer gets 0% chance).
    - Starting score (random recipe): About 1.1 for 3 types, good for checking code.
  - **SVM vs. Softmax**:
    - SVM: Wants right answer to beat wrong ones by 1; stops caring if it’s way better.
    - Softmax: Always wants right answer to have a higher chance, even if it’s already good.
    - SVM is chill with local wins; softmax looks at all pictures.
    - They work about the same, but people often pick softmax (not sure why).

## 6. Optimization

- **Technical Language**:
  - **Goal**: Minimize loss $L(W)$ by adjusting $W$ using gradient-based methods.
  - **Gradient**: Vector of partial derivatives $\nabla_W L$, indicating the direction of steepest loss increase.
  - **Numerical Gradient**:
    - Approximates gradient via finite differences: $\frac{L(W + h e_i) - L(W)}{h}$ for each dimension $i$.
    - Slow (requires evaluating loss for each of millions of parameters) and approximate.
  - **Analytic Gradient**:
    - Computes exact gradient using calculus (e.g., derivatives of SVM or softmax loss).
    - Fast, exact, but error-prone due to complex derivations.
  - **Practice**:
    - Use analytic gradient for optimization, verified by numerical gradient checks to ensure correctness.
    - Gradient check: Compare analytic and numerical gradients; they should match closely.
  - **Gradient Descent**:
    - Update rule: $W \gets W - \eta \nabla_W L$, where $\eta$ (learning rate) is a hyperparameter.
    - Negative gradient direction reduces loss.
  - **Mini-Batch Gradient Descent**:
    - Computes gradient on small subsets (e.g., 32, 64, 128 examples) for efficiency.
    - Noisy but allows more steps; batch size determined by GPU memory (e.g., 6-12 GB).
  - **Learning Rate**:
    - Critical hyperparameter; too high causes divergence, too low slows convergence.
    - Often starts high and decays over time to settle into minima.
  - **Variants**:
    - Stochastic Gradient Descent (SGD): Basic update.
    - Momentum: Tracks velocity, accelerating in consistent gradient directions.
    - RMSProp, Adam: Advanced methods for faster convergence.
  - **Random Search Baseline**: Sampling random $W$ yields ~15.5% accuracy on CIFAR-10 (vs. 10% chance, 95% state-of-the-art), highlighting need for gradient-based optimization.
  - **Loss Landscape**: High-dimensional space where loss is height; optimization seeks the lowest valley.

- **Simpler Language**:
  - **Goal**: Make our badness score as low as possible by tweaking our recipe ($W$) using math hints.
  - **Gradient**: A list of hints telling us which way to tweak each number in $W$ to make badness worse; we go the opposite way.
  - **Numerical Gradient**:
    - Guess the hints by slightly changing $W$, checking if badness goes up or down.
    - Super slow (need to check millions of numbers) and not perfect.
  - **Analytic Gradient**:
    - Use math (calculus) to get exact hints quickly.
    - Tricky to get right because math is hard.
  - **How We Do It**:
    - Use math hints, but double-check with guess hints to make sure we didn’t mess up.
    - Check: Both hints should be almost the same.
  - **Gradient Descent**:
    - Tweak $W$ a little: $W \gets W - \eta \times \text{hints}$, where $\eta$ (step size) decides how big the tweak is.
    - Go opposite the hints to lower badness.
  - **Mini-Batch**:
    - Only check badness on a few pictures (e.g., 32 or 64) at a time to go faster.
    - Hints are a bit messy but let us tweak more often; size depends on computer memory.
  - **Step Size**:
    - Super important; too big, and we jump around and fail; too small, and we take forever.
    - Start big, then make smaller to settle into a good recipe.
  - **Fancier Ways**:
    - Basic: Just use hints (SGD).
    - Momentum: Remember past hints to speed up in one direction.
    - RMSProp, Adam: Smarter tweaks to go faster.
  - **Random Guessing**: Picking random recipes gets 15.5% right on CIFAR-10 (better than 10% guessing, but way worse than 95% best), so we need hints.
  - **Picture It**: Badness is like a huge hilly land; we’re blindfolded, trying to find the lowest spot.

## 7. Interactive Demo

- **Technical Language**:
  - **Purpose**: Visualizes optimization for a 2D classification problem with three classes, each with three examples.
  - **Setup**:
    - Data: 2D points with labels; $W$ is a 3x2 matrix (rows for classifiers) plus biases.
    - Loss: Data loss (SVM or softmax) + L2 regularization; total loss displayed.
    - Visualization: Shows hyperplanes (score = 0), score increase directions, and loss per point.
  - **Features**:
    - Adjust $W$ and biases manually to see loss changes.
    - Parameter updates use gradients to reduce loss, visualized as hyperplane shifts.
    - Toggle between SVM and softmax loss; solutions are similar but differ slightly.
    - Step size controls update magnitude; randomization resets $W$.
  - **Insights**:
    - Demonstrates gradient descent converging to low loss.
    - Highlights regularization’s role in preferring diffuse weights.

- **Simpler Language**:
  - **What It Is**: A cool online tool to see how our recipe ($W$) learns to sort 2D dots into three types.
  - **How It Looks**:
    - Dots are in a 2D space, each with a type; $W$ is a small table plus extra tweaks (biases).
    - Shows badness score (SVM or softmax) plus niceness penalty; total badness is what we care about.
    - Draws lines where scores are 0 and arrows where scores get bigger.
  - **What You Can Do**:
    - Change $W$ or tweaks to see how badness changes.
    - Click to let math hints move lines to lower badness.
    - Switch between SVM and softmax; they’re close but a bit different.
    - Pick how big tweaks are; reset $W$ to random.
  - **What It Teaches**:
    - Shows how math hints help find a good recipe.
    - Niceness rule keeps the recipe balanced.

## 8. Historical Context: Pre-CNN Computer Vision

- **Technical Language**:
  - **Pre-2012 Pipeline**:
    - Feature extraction: Compute hand-engineered descriptors (e.g., color histograms, SIFT, HOG, LBP, textons) summarizing image properties (e.g., color distributions, edge orientations).
    - Bag-of-Words: Represent images as histograms over a dictionary of visual centroids (e.g., via k-means), capturing statistical patterns.
    - Concatenation: Combine multiple feature vectors into a large descriptor.
    - Classification: Apply linear classifier (e.g., SVM) on concatenated features.
  - **Limitations**:
    - Features were manually designed, limiting adaptability.
    - Linear classifiers on raw pixels fail to capture complex patterns (e.g., multiple object modes).
  - **Post-2012 Shift**:
    - End-to-end learning: Train a single differentiable model (e.g., CNN) from raw pixels to class scores.
    - Eliminates hand-engineered features, learning optimal feature extractors via backpropagation.
  - **Impact**: Significantly improved performance by optimizing the entire pipeline.

- **Simpler Language**:
  - **Old Way (Before 2012)**:
    - Step 1: Pull out key info from pictures, like color counts, edge directions, or patterns (e.g., color histograms, SIFT, HOG).
    - Step 2: Make a “word list” of common picture bits (like green edges) and count how many each picture has.
    - Step 3: Mash all this info into one big list.
    - Step 4: Use a math recipe (like SVM) to guess what’s in the picture based on this list.
  - **Problems**:
    - People had to guess what info mattered, which wasn’t always right.
    - Math recipes on raw picture pixels didn’t work well for tricky patterns.
  - **New Way (After 2012)**:
    - Use one big system (like a CNN) that learns everything from raw pictures to guesses.
    - No guessing what’s important; the system figures it out by learning from pixels.
  - **Why Better**: Works way better because the whole system learns together.

## 9. Upcoming Topics

- **Technical Language**:
  - **Backpropagation**: Efficient computation of analytic gradients for complex models.
  - **Neural Networks**: Introduction to multi-layer architectures for improved classification.

- **Simpler Language**:
  - **Backpropagation**: A fast way to get math hints for tweaking our recipe, even for complicated systems.
  - **Neural Networks**: Smarter systems with multiple layers to guess pictures better.

## 10. Key Takeaways

- **Technical Language**:
  - **Loss Functions**: SVM enforces margins; softmax maximizes correct class probability; both quantify classifier performance.
  - **Regularization**: L2 regularization prefers diffuse weights, improving generalization.
  - **Optimization**: Gradient descent (mini-batch) minimizes loss; learning rate and regularization strength are critical hyperparameters.
  - **Historical Shift**: From hand-engineered features and linear classifiers to end-to-end CNNs.
  - **Practical Tools**: Analytic gradients with numerical checks; interactive demos visualize optimization.

- **Simpler Language**:
  - **Loss Functions**: SVM wants right answers to beat wrong ones by 1; softmax wants right answers to have high chances.
  - **Niceness Rule**: Keeps recipe numbers small and spread out for better guesses on new pictures.
  - **Making It Better**: Use math hints to tweak the recipe; step size and niceness strength are super important.
  - **How It Changed**: Went from picking picture parts by hand to letting computers learn everything from pixels.
  - **Cool Stuff**: Use math to get hints, check them, and play with a tool to see how it works.
