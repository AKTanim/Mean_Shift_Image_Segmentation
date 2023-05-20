****** Image segmentation using mean-shift algorithm ******
------------------------------------------------------------
#########################################
There are two versions of the algorithm:
	1. Simple mean-shift
	2. Optimized mean-shift

All the experiment results similar to those in the report can be generated using the sollowing command line:

python main.py --image <image_path> -r <radius_value> -c <c_value> --feature_type <feature_type_value> --down_size_by <divisor to downsize> -experiment <experiment_type>

1. Replace <image_path> with the directory path of the image, e.g., 'Images\input\img1'.

2. Replace <radius_value> with an integer value for the parameter r.

3. Replace <c_value> with an integer value for the parameter c.

	* It is advised to use r = 2 and c = 4 for the pts.mat data
	** It is advised to use r = 10 and c = 4 for the images in the input folder

4. Replace <feature_type_value> with either '3D' or '5D' depending on which feature dimension is preferred.

5. Replace <divisor to downsize> with an integer value which will downscale the image by that value.
	
	* It is advised to use 2 to create images of half the shape of the original.

6. Replace <experiment_type> with one of the integer values from 0 to 4.
	
	* The experiment number creates the exact same output shown in the report.

------------------------------------------------------------------
Happy Colors