# Lecture 1: Introduction to Computer Vision and CS231n

These notes cover Lecture 1 of CS231n, providing an introduction to the course, the field of computer vision, its historical context, and the focus on deep learning, particularly convolutional neural networks (CNNs) for visual recognition. Each topic is explained in bullet points, first in **Technical Language** for advanced learners, then in **Simpler Language** for beginners.

## 1. Administrative Announcements

- **Technical Language**:
  - **Course Logistics**: CS231n is the second offering, with enrollment doubled from 180 to approximately 350 students.
  - **Video Recording**: Lectures are recorded; students uncomfortable with being filmed should position themselves outside camera range. Consent forms for recording will be distributed.
  - **Instructors**: Co-taught by Professor Fei-Fei Li, Andrej Karpathy, and Justin Johnson. Fei-Fei Li will deliver the first lecture, with Karpathy and Johnson handling most subsequent lectures due to Li’s upcoming maternity leave.
  - **Teaching Assistants (TAs)**: A team of TAs supports the course, introduced at the lecture’s end.
  - **Communication**: Piazza and the staff mailing list are the primary channels for course-related queries. Personal emails are reserved for confidential issues only.
  - **Late Policy**: Seven late days are provided for assignments, usable without penalty. Beyond this, penalties apply unless exceptional circumstances (e.g., medical or family emergencies) are discussed individually.
  - **Honor Code**: Strict adherence to Stanford’s honor code is expected. Collaboration policies must be followed to avoid academic integrity violations.

- **Simpler Language**:
  - **Class Size**: This class is way bigger than last time, with about 350 students instead of 180.
  - **Recording**: We’re filming the lectures. If you don’t want to be on camera, sit where the camera can’t see you. You’ll get a form to say it’s okay to be recorded.
  - **Teachers**: Professor Fei-Fei Li starts the class, but Andrej Karpathy and Justin Johnson will teach most of it because Fei-Fei is having a baby soon.
  - **Helpers**: There’s a team of teaching assistants (TAs) to help you, and you’ll meet them later.
  - **Talking to Us**: Use Piazza or the class email list for questions about the course. Only send personal emails for private stuff.
  - **Late Work**: You get seven extra days to turn in assignments without trouble. After that, you lose points unless something serious (like being sick) happens, and you talk to us.
  - **Be Honest**: Follow Stanford’s rules about not cheating. Work together only as allowed, or you could get in big trouble.

## 2. Course Overview and Scope

- **Technical Language**:
  - **Course Focus**: CS231n is a computer vision course emphasizing neural networks, specifically convolutional neural networks (CNNs), for visual recognition tasks, with a primary focus on image classification.
  - **Relation to Other Courses**:
    - CS131 (Introduction to Computer Vision) is not a strict prerequisite but provides useful background. Students without prior vision knowledge should review CS131 notes.
    - CS231a (graduate-level course by Professor Silvio Savarese) covers broader computer vision topics, including 3D vision and robotics, and complements CS231n without significant overlap.
  - **Learning Objectives**: Provide an in-depth understanding of deep learning models (CNNs) and their application to visual recognition, equipping students with hands-on skills to implement state-of-the-art algorithms.
  - **Curriculum**: Focuses on recent advancements (up to 2015), including practical projects (e.g., style transfer) and theoretical foundations of deep learning.
  - **Philosophy**: Emphasizes hands-on learning, exposure to cutting-edge material, and practical coding skills for building deep learning systems.

- **Simpler Language**:
  - **What’s This Class About**: This class teaches how computers understand pictures using special math models called neural networks, especially convolutional neural networks (CNNs). We’ll mostly work on figuring out what’s in a picture (image classification).
  - **Other Classes**:
    - CS131 is a beginner vision class. You don’t have to take it first, but it helps. If you’re new, check its notes to catch up.
    - CS231a is another class about vision, covering things like 3D and robots. It’s different but related, so you can take both.
  - **What You’ll Learn**: You’ll dive deep into CNNs, learn how to code them, and build cool projects with the latest tech (stuff from 2015).
  - **How We Teach**: You’ll do lots of coding and projects, like turning pictures into art, so you really get how this stuff works.
  - **Our Goal**: Make you great at building vision systems and understanding the newest ideas.

## 3. Importance of Computer Vision

- **Technical Language**:
  - **Data Explosion**: By 2016, Cisco estimated that over 85% of internet data consists of multimedia (pixels), driven by the proliferation of sensors (e.g., smartphones, cameras, autonomous vehicles).
  - **Challenge**: Visual data is the “dark matter” of the internet—abundant but difficult to process due to its volume and complexity. For example, YouTube uploads 150 hours of video every 60 seconds, infeasible for human annotation.
  - **Applications**: Computer vision enables automated labeling, indexing, and content management for platforms like YouTube, supporting tasks like object recognition, search, and advertisement placement.
  - **Interdisciplinary Nature**: Computer vision intersects with engineering, physics, biology, psychology, computer science, mathematics, and fields like NLP, robotics, and medical imaging.
  - **Research Context**: Fei-Fei Li’s Stanford Computer Vision Lab focuses on machine learning, deep learning, cognitive science, neuroscience, and NLP-speech intersections.

- **Simpler Language**:
  - **Tons of Pictures**: Most internet stuff (over 85% by 2016) is pictures or videos, thanks to phones, cameras, and even cars with cameras.
  - **Big Problem**: There’s so much picture data (like 150 hours of YouTube videos every minute) that people can’t label it all. It’s like “dark matter”—we have it, but it’s hard to understand.
  - **Why It Matters**: Vision tech helps computers figure out what’s in pictures or videos, like finding a basketball shot or putting ads in the right place.
  - **Lots of Fields**: Vision connects to science, math, biology, psychology, robots, and more, so it’s a mix of many subjects.
  - **Research**: Fei-Fei Li’s lab at Stanford works on teaching computers to understand pictures, think like brains, and even talk about what they see.

## 4. Brief History of Computer Vision

- **Technical Language**:
  - **Evolutionary Origins (540M Years Ago)**:
    - Cambrian Explosion: Andrew Parker’s theory posits that the development of simple eyes (e.g., trilobite pinhole cameras) triggered a biological arms race, driving speciation by enabling predation and evasion.
    - Vision became a key evolutionary driver, necessitating complex visual processing systems.
  - **Camera Obscura (Renaissance)**:
    - Leonardo da Vinci documented the camera obscura, a device using a lens or pinhole to project real-world scenes, marking the start of engineered vision for duplicating visual information.
  - **Neuroscience Breakthrough (1950s-60s)**:
    - Hubel and Wiesel’s Nobel Prize-winning work (1981) revealed that the primary visual cortex processes simple edge-like structures, not holistic objects, using electrodes in cat brains.
    - Implication: Visual processing is hierarchical, starting with low-level features, foundational for deep learning architectures.
  - **Computer Vision Beginnings (1960s)**:
    - Larry Roberts’ 1963 dissertation on edge extraction in “block world” laid groundwork for computer vision, focusing on shape-defining edges.
    - MIT’s 1966 Summer Vision Project aimed to solve vision in one summer, marking the field’s formal start, though it underestimated the challenge.
  - **Hierarchical Models (1970s)**:
    - David Marr’s book *Vision* proposed a hierarchical visual processing model: primal sketch (edges), 2.5D sketch (depth cues), and 3D model for navigation/manipulation.
    - Influenced early recognition algorithms, e.g., Tom Binford’s generalized cylinder model and SRI’s pictorial structure model, which used simple shapes and probabilistic parts.
  - **Perceptual Grouping (1990s)**:
    - Normalized Cut (Jianbo Shi, Jitendra Malik) addressed segmenting real-world images into meaningful parts, a fundamental unsolved problem.
  - **Face Detection (2000s)**:
    - Viola-Jones face detector (2001) used learned features for real-time face detection, deployed in Fujifilm cameras by 2006, shifting focus from 3D modeling to recognition.
  - **Feature-Based Recognition (2000s)**:
    - SIFT (David Lowe) introduced robust feature detection, enabling object recognition across angles and clutter, dominating the field until deep learning’s resurgence.
  - **Benchmarks and Datasets (2000s-2010s)**:
    - PASCAL VOC (2005-2012) standardized object recognition with 20 classes.
    - ImageNet (2010-present), created by Fei-Fei Li’s lab, scaled to 50 million images and 20,000 classes, using Amazon Mechanical Turk for annotations.
  - **Deep Learning Revolution (2012)**:
    - Alex Krizhevsky and Geoff Hinton’s CNN (AlexNet) won the 2012 ImageNet Challenge, halving error rates using a high-capacity, end-to-end trained model.
    - Built on earlier work (e.g., Fukushima’s Neocognitron, LeCun’s CNNs for digit recognition), with advances in hardware (GPUs) and data availability.

- **Simpler Language**:
  - **Vision in Nature (540M Years Ago)**:
    - A long time ago, animals like trilobites got simple eyes, starting a race where animals had to see to hunt or escape, making life way more complex.
    - Eyes changed how animals evolved, making vision super important.
  - **Early Cameras (Renaissance)**:
    - Leonardo da Vinci wrote about the camera obscura, a box with a hole that copies the outside world onto a surface, starting the idea of building vision tools.
  - **Brain Science (1950s-60s)**:
    - Scientists Hubel and Wiesel stuck tiny needles in cat brains and found that the brain sees simple lines and edges first, not whole things like fish. This won a big prize.
    - This idea—that vision starts small and builds up—helped create deep learning.
  - **Computer Vision Starts (1960s)**:
    - Larry Roberts wrote a paper in 1963 about finding edges in block pictures, kicking off computer vision.
    - In 1966, MIT thought they could solve vision in one summer. They didn’t, but it started the field.
  - **Layered Vision (1970s)**:
    - David Marr said vision works in steps: first edges, then some depth, then a full 3D picture to move around in the world.
    - Early ideas at Stanford used simple shapes (like cylinders) or parts (like eyes and nose) to recognize things.
  - **Splitting Pictures (1990s)**:
    - A project called Normalized Cut tried to divide real pictures into parts (like heads or chairs), but it’s still a hard problem we haven’t fully solved.
  - **Finding Faces (2000s)**:
    - The Viola-Jones system (2001) learned to spot faces in any picture super fast, ending up in cameras by 2006. It focused on recognizing, not building 3D shapes.
  - **Key Features (2000s)**:
    - SIFT (by David Lowe) found important spots in pictures (like corners) to recognize things even if they’re turned or messy. This ruled vision for years.
  - **Testing Progress (2000s-2010s)**:
    - PASCAL VOC tested vision systems on 20 things (like dogs or planes).
    - ImageNet, made by Fei-Fei Li’s team, used 50 million pictures with 20,000 things, labeled by online workers, to push vision forward.
  - **Deep Learning Boom (2012)**:
    - In 2012, Alex Krizhevsky and Geoff Hinton’s CNN (AlexNet) crushed the ImageNet contest, cutting errors in half with a system that learned everything itself.
    - This used old ideas (from Fukushima and LeCun) but worked better because of faster computers (GPUs) and tons of pictures.

## 5. Convolutional Neural Networks (CNNs)

- **Technical Language**:
  - **Architecture**: CNNs are a type of deep learning model inspired by biological vision (Hubel and Wiesel) and early models like Fukushima’s Neocognitron and LeCun’s digit-recognizing CNNs.
  - **Historical Context**:
    - Yann LeCun’s 1990s CNNs, developed at Bell Labs, recognized digits for zip codes and checks, using hierarchical feature extraction (edges to complex patterns).
    - AlexNet (2012) scaled this architecture, leveraging GPU acceleration and large datasets (ImageNet), with minor modifications (e.g., ReLU activation instead of sigmoid).
  - **Key Advances**:
    - Hardware: Moore’s Law and Nvidia GPUs enabled training of high-capacity models, overcoming computational bottlenecks.
    - Data: ImageNet’s scale (1.5M images, 1,000 classes) reduced overfitting and supported end-to-end training.
  - **Evolution**: By 2015, CNNs like Microsoft’s 151-layer Residual Net dominated ImageNet, showing continued architectural refinement.
  - **Significance**: CNNs shifted the field from hand-engineered features (e.g., SIFT, SVMs) to learned features, confirming the efficacy of hierarchical, data-driven models.

- **Simpler Language**:
  - **What’s a CNN**: A CNN is a smart math system that learns to see pictures like our brains do, starting with simple lines and building up to whole objects.
  - **Where It Came From**:
    - In the 1990s, Yann LeCun made CNNs to read zip codes and checks, looking at edges first, then bigger patterns.
    - In 2012, AlexNet (by Krizhevsky and Hinton) used a similar idea but made it bigger and better, winning ImageNet.
  - **Why It Got Better**:
    - Faster Computers: New chips (GPUs) made CNNs quick enough to handle big systems.
    - More Pictures: ImageNet gave millions of pictures to learn from, so CNNs didn’t make as many mistakes.
  - **What’s New**: By 2015, CNNs got super deep (151 layers) and kept winning contests, getting smarter every year.
  - **Why It’s Cool**: CNNs learn to see by themselves, unlike older systems where people had to pick what’s important in pictures.

## 6. Visual Recognition and Beyond

- **Technical Language**:
  - **Primary Task**: CS231n focuses on image classification, a core visual recognition problem, but introduces related tasks (e.g., object detection, image captioning).
  - **Task Definitions**:
    - Image Classification: Assign a single label to an entire image.
    - Object Detection: Localize and classify objects within an image (e.g., bounding boxes for cars, pedestrians).
    - Image Captioning: Generate descriptive text for an image, requiring scene understanding.
  - **Broader Challenges**:
    - Perceptual Grouping: Segmenting scenes into meaningful parts remains unsolved.
    - 3D Integration: Combining recognition with 3D modeling for robotics and navigation.
    - Scene Understanding: Projects like Visual Genome aim to capture relationships and narratives in images, moving beyond labeling to storytelling.
  - **Holy Grails**:
    - Narrative Generation: Enable computers to describe a scene’s story from a single image, akin to human perception (e.g., 500ms glance yields detailed descriptions).
    - Nuanced Understanding: Capture social, emotional, and contextual nuances in images (e.g., humor, interactions), as exemplified by complex scenes like Obama playing ping-pong.
  - **Applications**: Image classification underpins commercial tasks (e.g., online shopping, photo organization) and societal benefits (e.g., medical imaging, autonomous robots).

- **Simpler Language**:
  - **Main Job**: This class is mostly about teaching computers to name what’s in a picture (image classification), but we’ll also touch on other jobs like finding objects or describing pictures.
  - **What’s What**:
    - Image Classification: Say one thing about the whole picture (e.g., “cat”).
    - Object Detection: Point out where things are, like circling a car or person.
    - Image Captioning: Write a sentence about the picture, like “A dog runs in a park.”
  - **Bigger Problems**:
    - Splitting Pictures: Figuring out what parts of a picture go together (like heads or chairs) is still super hard.
    - 3D Stuff: Mixing vision with 3D for robots to move around.
    - Understanding Scenes: Projects like Visual Genome try to explain what’s happening in a picture, like who’s doing what.
  - **Big Dreams**:
    - Tell Stories: We want computers to look at a picture for half a second and write a story about it, like humans can.
    - See Everything: Understand funny or social stuff in pictures, like Obama joking while playing ping-pong.
  - **Why It’s Useful**: This stuff helps with shopping online, sorting photos, or even building robots that save lives.

## 7. Key Takeaways

- **Technical Language**:
  - **Course Structure**: CS231n focuses on CNNs for image classification, with hands-on projects and exposure to 2015 advancements, supported by a rigorous honor code and late policy.
  - **Historical Context**: Computer vision evolved from biological vision (Cambrian Explosion), engineered systems (camera obscura), and neuroscience (Hubel and Wiesel) to modern CNNs (2012 ImageNet).
  - **Field Significance**: Vision handles the internet’s “dark matter” (pixel data), intersecting multiple disciplines and driving applications from AI to robotics.
  - **Deep Learning**: CNNs, built on decades of foundational work, leverage hardware and data to achieve state-of-the-art performance, shifting from engineered to learned features.
  - **Future Challenges**: Beyond classification, vision aims for scene understanding, 3D integration, and narrative generation, addressing unsolved problems like perceptual grouping.

- **Simpler Language**:
  - **Class Plan**: You’ll learn CNNs by coding and doing projects with the latest tech, following strict rules about cheating and late work.
  - **Vision’s Story**: Vision started with animal eyes, grew with cameras and brain science, and exploded with CNNs in 2012, thanks to fast computers and lots of pictures.
  - **Why It’s Big**: Vision deals with tons of internet pictures, mixing science, math, and more to help with robots, medicine, and apps.
  - **CNNs Rule**: They learn to see by themselves, getting better because of better computers and more data, building on old ideas.
  - **What’s Next**: We want computers to understand whole scenes, make 3D models, and tell stories, solving tricky problems like splitting up pictures.
