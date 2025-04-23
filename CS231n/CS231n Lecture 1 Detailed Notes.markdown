# CS231n Lecture 1: Introduction to Computer Vision and Deep Learning

## Course Overview
- **Course**: CS231n - Deep Learning and Neural Networks for Visual Recognition
- **Instructors**: 
  - Professor Fei-Fei Li (Computer Science Department, Stanford)
  - Co-taught with senior graduate students Andrej Karpathy and Justin Johnson
- **Enrollment**: Approximately 350 students, doubled from the previous offering (180 students).
- **Recording**: Lectures are video-recorded. Students uncomfortable with being recorded can sit behind the camera or in corners not covered by it. Consent forms will be provided.
- **Teaching Assistants (TAs)**: A team of TAs will be introduced later in the lecture.

### Course Logistics
- **Communication**:
  - Primary channels: Piazza and staff mailing list for course-related queries.
  - Personal emails only for confidential issues (e.g., personal or medical emergencies).
  - Professor Li will be on maternity leave starting late January, so responses may be delayed.
- **Philosophy**:
  - Hands-on, project-based learning.
  - Exposure to state-of-the-art material (up to 2015 research).
  - Practical skills in building deep learning code.
  - Fun elements like generating artistic images (e.g., Van Gogh-style transformations).
- **Grading Policies**:
  - Available on the course website.
  - **Late Policy**: 7 late days allowed with no penalty. Beyond that, penalties apply unless exceptional circumstances (e.g., medical or family emergencies) are discussed individually.
  - **Honor Code**: Strict adherence expected. Collaboration policies must be followed. Violations will be taken seriously.
- **Prerequisites**:
  - Basic understanding of computer vision recommended (e.g., CS131).
  - No strict requirement, but students new to computer vision should catch up using provided notes.

## Introduction to Computer Vision
- **Definition**: Computer vision is a field focused on enabling computers to interpret and understand visual data (images and videos) using neural networks, particularly convolutional neural networks (CNNs).
- **Significance**:
  - Vision is one of the fastest-growing areas in artificial intelligence (AI).
  - Cisco estimates that by 2016, over 85% of internet data was multimedia (pixels), termed the "dark matter" of the internet due to its volume and difficulty to process.
- **Challenges**:
  - **Data Volume**: YouTube uploads 150+ hours of video every 60 seconds, making manual annotation impossible.
  - **Data Complexity**: Visual data is hard to harness compared to text or structured data.
  - **Applications**: Indexing, searching, managing, and monetizing visual content (e.g., advertisements).

### Why the Explosion of Visual Data?
- **Internet**: Acts as a carrier for massive data dissemination.
- **Sensors**: Proliferation of smartphones, digital cameras, and car-mounted cameras has led to more sensors than people on Earth.

### Interdisciplinary Nature
- Computer vision intersects with:
  - Engineering
  - Physics
  - Biology
  - Psychology
  - Computer Science
  - Mathematics
- Relevant fields include:
  - Machine learning
  - Cognitive science
  - Neuroscience
  - Natural language processing (NLP)
  - Speech processing
  - Robotics
  - Medical imaging

## Stanford’s Computer Vision Curriculum
- **CS131**: Introductory computer vision course (offered previous quarter).
- **CS231n**: Focuses on deep learning and neural networks for visual recognition, particularly CNNs.
- **CS231a**: Graduate-level course on broader computer vision topics (e.g., 3D vision, robotics) taught by Professor Silvio Savarese. Offered next quarter.
  - **Difference**: CS231n is more specialized (neural networks, visual recognition), while CS231a covers broader topics. They complement rather than replace each other.
- **Advanced Courses**: Potential 700-level courses in the planning stage (check syllabus for updates).

## Brief History of Computer Vision
### Evolutionary Perspective (540 Million Years Ago)
- **Cambrian Explosion**:
  - Around 540 million years ago, a period called the "Big Bang of evolution" saw a rapid diversification of species (speciation).
  - Trigger: Development of the eye in trilobites (simple pinhole camera-like structure).
  - **Impact**: Vision enabled predators to locate prey, sparking a biological arms race (predators vs. prey developing survival mechanisms).
  - Vision became a major driver of evolution.

### Renaissance: Camera Obscura
- **Leonardo da Vinci** (circa 1500s):
  - Documented the **camera obscura**, a device using a lens or hole to project light from the real world onto a surface, capturing visual information.
  - Marked the beginning of engineered vision, focused on duplicating the visual world (not understanding it).
- **Post-Renaissance**:
  - Development of film (e.g., Kodak’s commercial cameras).
  - Introduction of camcorders.

### Neuroscience: Vision in the Brain
- **Hubel and Wiesel (1950s-1960s)**:
  - Studied the **primary visual cortex** in cats using electrodes.
  - **Key Finding**: Neurons in the primary visual cortex respond to simple oriented edges (not holistic objects like fish or mice).
  - **Implications**:
    - Vision processing starts with simple structures (edges).
    - Influenced both neuroscience and engineering models (e.g., deep learning features resemble these edge-like structures).
  - **Recognition**: Won the Nobel Prize in Medicine (1981).
  - **Note**: The primary visual cortex is located at the back of the brain, far from the eyes, unlike other sensory cortices (e.g., olfactory, auditory). Nearly 50% of the brain is involved in vision, highlighting its complexity and importance.

### Birth of Computer Vision
- **Larry Roberts (1963)**:
  - PhD dissertation on the "block world," extracting edge-like structures from images to recognize blocks under varying lighting and orientations.
  - Considered a precursor to modern computer vision.
  - Roberts later contributed to the internet’s development at DARPA.
- **Summer of 1966**:
  - MIT AI Lab launched the **Summer Vision Project**, aiming to solve vision in one summer.
  - Outcome: Vision was not solved, but this marked the formal birth of computer vision as a field.
  - **Context**: AI labs were established at MIT (Marvin Minsky) and Stanford (John McCarthy, who coined "artificial intelligence") in the early 1960s.

### David Marr’s Contribution (1970s)
- **Vision Book**: Proposed that vision is **hierarchical**.
  - **Primal Sketch**: Initial stage focusing on edges (inspired by Hubel and Wiesel).
  - **2.5D Sketch**: Reconciles 2D images with the 3D world (e.g., inferring depth and occlusion).
  - **3D Model**: Full reconstruction for navigation and manipulation.
- **Significance**: Laid the foundation for hierarchical models like CNNs, though Marr didn’t specify mathematical or learning procedures.

### Early Visual Recognition Models (1970s-1980s)
- **Generalized Cylinder Model (Tom Binford, Rodney Brooks, Stanford)**:
  - Objects are combinations of simple shapes (cylinders, blocks) viewed from different angles.
  - Influential in the 1970s.
- **Pictorial Structure Model (Stanford Research Institute)**:
  - Objects composed of parts (e.g., head = eyes + nose + mouth) connected by springs, allowing deformation.
  - Introduced variability in recognition.

### Perceptual Grouping (1990s)
- **Normalized Cut (Jitendra Malik, Stanford/Berkeley)**:
  - Addressed segmenting real-world color images into sensible parts (e.g., grouping heads, chairs).
  - A fundamental, unsolved problem in vision.
- **Shift to Real-World Images**: Moved from black-and-white or synthetic images to colorful, real-world data.

### Face Detection (2000s)
- **Viola-Jones Face Detector (Paul Viola, Michael Jones)**:
  - Learned simple black-and-white filter features to detect faces in the wild.
  - First real-time computer vision algorithm (ran on Pentium 2 chips).
  - Deployed in Fujifilm’s 2006 smart digital camera, marking rapid technology transfer.
  - Shifted focus from 3D modeling to recognition.

### Feature-Based Recognition
- **SIFT (Scale-Invariant Feature Transform, David Lowe)**:
  - Identified key features on objects for recognition across angles and cluttered scenes.
  - Dominated computer vision from 2000–2012.
  - Deep learning later confirmed the importance of similar learned features.

### Machine Learning Tools
- **Pre-Deep Learning**:
  - Graphical models and support vector machines (SVMs) were common.
  - Used for scene recognition and object detection.
- **Deformable Part Model (2009-2010)**:
  - Learned object parts and their spatial configurations using SVMs.
  - Applied to real-world problems like pedestrian and car detection.

### Benchmarking
- **PASCAL VOC Challenge (2000s)**:
  - European effort with tens of thousands of images across 20 object classes (e.g., dogs, cows, airplanes).
  - Annual competitions improved performance.
- **ImageNet (2010)**:
  - Created by Fei-Fei Li’s lab using Amazon Mechanical Turk.
  - 50 million images, 20,000+ object classes.
  - **ImageNet Challenge**: 1.5 million images, 1,000 classes.
  - Called the “Olympics of computer vision.”

### Deep Learning Revolution
- **2012 ImageNet Challenge**:
  - **AlexNet** (Alex Krizhevsky, Geoff Hinton): A 7-layer CNN won by a large margin, cutting error rates significantly.
  - **Significance**: Marked the deep learning revolution, covered by major media (e.g., New York Times).
  - **Note**: CNNs were not new (originated in the 1970s-1980s) but gained prominence due to:
    - **Hardware**: GPUs (Nvidia) and Moore’s Law enabled faster training of large models.
    - **Data**: Large datasets like ImageNet prevented overfitting and supported end-to-end training.
- **Historical Foundations**:
  - **Kunihiko Fukushima**: Developed the neocognitron, an early neural network model.
  - **Yann LeCun (1990s)**: Built CNNs at Bell Labs for digit recognition (e.g., zip codes, checks). Inspired by Hubel and Wiesel’s edge-based processing.
  - **Backpropagation**: Developed in the 1980s-1990s by Geoff Hinton and others, enabling neural network training.
- **Post-2012**:
  - CNNs dominated ImageNet (e.g., 151-layer ResNet by Microsoft Asia in 2015).
  - Models grew in capacity, with tweaks like rectified linear units (ReLUs) replacing sigmoids.

## CS231n Focus
- **Primary Task**: Image classification (assigning labels to entire images).
- **Other Tasks** (covered later):
  - Object detection (locating objects).
  - Image captioning (describing images).
  - Dense labeling and perceptual grouping.
  - 3D modeling and robotics.
  - Motion and affordance analysis.
- **Architecture**: Emphasis on CNNs, the most successful deep learning model for vision.
- **Holy Grails**:
  - **Scene Storytelling**: Describing a scene in detail (e.g., essays from 500ms image exposure).
  - **Nuanced Understanding**: Capturing humor, interactions, and relationships (e.g., Visual Genome project).

## Applications of Image Classification
- **Commercial**: Object recognition for online/mobile shopping, album sorting.
- **Startups**: Food recognition, product identification.
- **Industry**: Robotics, medical imaging, exploration.

## Broader Visual Intelligence
- Vision extends beyond recognition to:
  - Navigation and manipulation (e.g., robotics).
  - Socializing and entertainment (e.g., understanding humor).
  - Learning and understanding the world.
- **Impact**: Computer vision enhances robotics, saves lives, and enables exploration.

## Study Tips
- **Review Prerequisites**: Brush up on CS131 notes if new to computer vision.
- **Engage with Piazza**: Stay active for clarifications and updates.
- **Hands-On Practice**: Focus on coding assignments to master deep learning implementation.
- **Understand History**: Contextualize CNNs within the evolution of vision (biological and engineered).
- **Key Concepts**:
  - Hierarchical processing (Marr, Hubel-Wiesel).
  - Feature-based recognition (SIFT, Viola-Jones).
  - Importance of data and hardware in deep learning’s success.
- **Prepare for Midterm**: Sample midterms will be provided (details TBD).