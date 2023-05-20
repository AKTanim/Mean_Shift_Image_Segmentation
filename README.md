# Mean_Shift_Image_Segmentation
### Efficacy of mean-shift algorithm for image segmentation

This project was a part of my Computer Vision course at Maastricht University. It is an implementation and experimentation of the Mean-Shift algorithm for image segmentation. The following steps are performed:

    1. Implementation of the simple Mean-Shift algorithm and testing it on given test data.
    2. Optimization of the algorithm using two speedup methods and testing it on the test data.
    3. Implementing the Mean-Shift algorithm on images, considering the suggested optimizations.
    4. Applying the algorithm on image features, transforming the result back to an image, and visualizing the obtained segmentation.
    5. Testing different parameters such as radius (r), parameter (c), and feature types.

<p align="center">
  <img src="https://github.com/AKTanim/Mean_Shift_Image_Segmentation/blob/main/Images/output/exp_1/img1_0.png" alt="Original Image">
  <img src="https://github.com/AKTanim/Mean_Shift_Image_Segmentation/blob/main/Images/output/exp_1/img1_1.png" alt="Segmented image">
</p>

The [images](Images/input) are from the Berkeley Segmentation Dataset: [image 1](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/dataset/images/color/181091.html), [image 2](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/dataset/images/color/55075.html), [image 3](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/dataset/images/color/368078.html)

The [report](Report.pdf) includes multiple experiments and their results:

### Experiment 0:
  Comparing the simple Mean-Shift algorithm with the optimized algorithm using two speedup methods. The optimized algorithm significantly reduces the computational cost.
### Experiment 1:
  Comparing the simple and optimized algorithms on images. The optimized algorithm reduces computational load through speedup methods.
### Experiment 2:
  Analyzing the influence of the radius (r) parameter on segmentation results. There is a sweet spot where segmentation is effective.
### Experiment 3:
  Analyzing the influence of the parameter (c) on segmentation results. The parameter controls a speedup method and affects the number of peaks.
### Experiment 4:
  Comparing the 3D and 5D feature spaces. Higher values of r are required for better segmentation in the 5D feature space.

*Please read the ['readme_code.txt'](readme_code.txt) file to run the ecperiments from the terminal. Specific shell commands are given in the [report](Report.pdf) as well.*

The report also proposes additional processing steps to improve segmentation results, including pre-processing for image enhancement, filtering for noise reduction, adaptive parameter selection, post-processing for refinement and smoothing, and incorporating prior knowledge.

Overall, the project presents a thorough analysis of the Mean-Shift algorithm for image segmentation, explores parameter influence, and suggests steps for further improvements.
