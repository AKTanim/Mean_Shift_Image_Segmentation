import scipy.io
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from skimage import color
from skimage.transform import resize
import time
import argparse

#--------------------------------------------------------------------------------------------------------
# Simple algorithm starts

def load_data():
    mat = scipy.io.loadmat('data\pts.mat') # loads .mat format
    data = mat.get('data') # takes the data array only
    print("Shape of data: {}".format(np.shape(data)))
    df = pd.DataFrame(data)
    display(df) # Displays data in a pretty format
    return data


def findpeak(data, idx, r):
    peak = data[idx].copy() #centroid is initiated at one of the datapoints
    global iterations  # to calculate the number of times find_peak_opt is called
    while True:
        '''
        for i, point in enumerate(data):
            if eucld_distance(peak, point) <= r:
                within_window.append(i)

        # I replace the above for loop with the following command.
        # It makes the code way lot faster.
        '''
        # calculate the Euclidean distance between the peak and iterating over all data points
        # and record the indices of the points which are inside the window i.e., < r
        within_window = np.where(np.linalg.norm(data - peak, axis=1) <= r)[0]
        new_peak = np.mean(data[within_window], axis=0) # mean of the points inside the window

        if np.linalg.norm(peak - new_peak, axis=0) < 0.01: # use the threshold 0.01
            # findpeak is terminated and already found peak is returned
            break
        peak = new_peak # new peak is updated
        iterations += 1
    return peak

def meanshift(data, r):
    n, p = data.shape # n = no of rows, p = no of columns
    labels = np.zeros(n, dtype=int) # we need labels for all the data points i.e., rows
    peaks = []
    for idx in range(n):
        if labels[idx] == 0:
            peak = findpeak(data, idx, r) # finds the peak for the data point data[idx]
            merged = False
            for i, existing_peak in enumerate(peaks):
                # Check whether found peak already exists
                # consider peaks similar if the distance between them is less than r/2
                if np.linalg.norm(peak - existing_peak, axis=0) < r / 2:
                    labels[idx] = i + 1 # label of existing peak is assigned to the new peak
                    merged = True
                    break

            if not merged:
                peaks.append(peak) #new peak is appended to the peaks list
                labels[idx] = len(peaks) #unique labels are created as the length of the peaks
    return labels, np.array(peaks)

def plot_clusters(data, labels, peaks, r):
    fig = plt.figure()  # Creating figure
    ax = fig.add_subplot(projection="3d")  # 3D plot

    # Scatter plot with different colors based on number of labels
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=1, alpha=0.3, c=labels)
    # Plot the centroids on the same plot
    ax.scatter(peaks[:, 0], peaks[:, 1], peaks[:, 2], s=60, color='black', alpha=1, marker='o')

    plt.title(f"Clusters (r = {r})")
    plt.show()


def test_algo(r):
    mat = load_data() # load the test data
    data = np.transpose(mat) # transpose the data from (3, 2000) to (2000, 3)
    print("Transposed data:", data.shape)
    labels, peaks = meanshift(data, r) # call the algorithm with r = 2

    # Plot the data
    plot_clusters(data, labels, peaks, r)
    print("Label shape: ", labels.shape)
    print("Unique Labels: ", set(labels))
    print("Peaks shape :", peaks.shape)
    print("Peaks :", peaks)

# simple algorithm ends
#---------------------------------------------------------------------------------------------------------------
# Optimized algorithm starts

def find_peak_opt(data, idx, r, threshold, c=4):
    peak = data[idx].copy() #centroid is initiated at one of the datapoints
    path = [peak] # needed when we want to draw the search path
    cpts = np.zeros(data.shape[0], dtype=int) # variable needed to implement the speedups
    global iterations # to calculate the number of times find_peak_opt is called

    while True:
        # calculate the Euclidean distance between the peak and iterating over all data points
        # and record the indices of the points which are inside the window i.e., < r
        within_window = np.where(np.linalg.norm(data - peak, axis=1) <= r)[0]
        new_peak = np.mean(data[within_window], axis=0) # mean of the points inside the window
        if np.linalg.norm(peak - new_peak, axis=0) < threshold: # use the threshold 0.01
            # find_peak_opt will be terminated
            # speedup 1: basin of attraction at final peak (distance < r)
            # cpts values of those points inside r are assigned to 1
            dist_to_path = np.linalg.norm(data - peak, axis=1)
            cpts[np.where(dist_to_path <= r)] = 1
            break
        path.append(new_peak)

        # speedup 2: search path's basin of attraction (distance < r/c)
        dist_to_path = np.linalg.norm(data - new_peak, axis=1)
        cpts[np.where(dist_to_path <= r / c)] = 1

        peak = new_peak # new peak is updated
        iterations += 1

    return peak, cpts

def meanshift_opt(data, r, c):
    n, p = data.shape # n = no of rows, p = no of columns
    labels = np.zeros(n, dtype=int) # we need labels for all the data points i.e., rows
    peaks = []
    for idx in range(n):
        if labels[idx] == 0:
            # finds the peak and cpts for the data point data[idx]
            peak, cpts = find_peak_opt(data, idx, r, threshold=0.01, c=c)
            merged = False
            for i, existing_peak in enumerate(peaks):
                # Check whether found peak already exists
                # consider peaks similar if the distance between them is less than r/2
                if np.linalg.norm(peak - existing_peak, axis=0) < r / 2: # label of existing peak is assigned to the new peak
                    labels[idx] = i + 1
                    labels[cpts == 1] = i + 1 # all the points with cpts =1 are assigned to the same peak (speedups)
                    merged = True
                    break

            if not merged:
                peaks.append(peak) #new peak is appended to the peaks list
                print("Number of peaks:", len(peaks))
                labels[cpts == 1] = len(peaks) #unique labels are created as the length of the peaks
    return labels, np.array(peaks)

def test_algo_opt(r):
    mat = load_data() # load the test data
    data = np.transpose(mat) # transpose the data from (3, 2000) to (2000, 3)
    print("Transposed data:", data.shape)
    labels, peaks = meanshift_opt(data, r, 4) # call the algorithm with r = 2 and c = 4

    # Plot the data
    plot_clusters(data, labels, peaks, r)
    print("Label shape: ", labels.shape)
    print("Unique Labels: ", set(labels))
    print("Peaks shape :", peaks.shape)
    print("Peaks :", peaks)

# Optimized algorithm ends
#-----------------------------------------------------------------------------------------------------------------
# Implementation on images start

def plot_images(axs, i, im, segments, runtime, r, c):
    axs.imshow(im)
    if segments != 0: # plot segmented image
        axs.set_title(f'Segmented Image (r = {r}, c = {c}')
    else: # plot original image
        axs.set_title('Original Image')

    axs.axis('off')
    axs.text(0, im.shape[0] + 10,
                f"Segments: {segments}, Iterations: {iterations}, Runtime: {runtime:.2f}s",
                color='black', fontsize=8, ha='left')
    plt.tight_layout()
    plt.show()

def segmIm(im, r, c, exp, feature_type):
    lab_image = color.rgb2lab(im) # color space conversion to CIELAB
    print("Original Image shape: ", lab_image.shape)
    n, m, _ = lab_image.shape # n = row, m = col, _ = RGB

    feature_matrix = lab_image.reshape((n * m, 3))  # Flattening the lab_image into a matrix
    print("Reshaped Image shape: ", feature_matrix.shape)

    if feature_type == '5D':
        # Pre-processing: Including spatial position information
        x_coords = np.tile(np.arange(n), m).reshape((n * m, 1))
        y_coords = np.repeat(np.arange(m), n).reshape((n * m, 1))

        # Concatenating the color channels and x, y coordinates
        feature_matrix = np.hstack((feature_matrix, x_coords, y_coords))

    if exp == 1: # Experiment 1: Image with simple vs optimized algorithms
        # initiate runtime for simple algorithm
        start_time_1 = time.time()

        # call the simple algorithm with r and c = 4
        labels, peaks1 = meanshift(feature_matrix, r)
        # Converting the resulting cluster centers back to RGB color space
        rgb_peaks = color.lab2rgb(peaks1[:, :3])
        segmented_image = rgb_peaks[labels - 1]  # Assigning colors to each pixel based on the labels
        segmented_image_1 = segmented_image.reshape((n, m, 3))  # Reshaping back to the original image dimensions
        runtime_1 = time.time() - start_time_1

        start_time_2 = time.time()
        # call the optimized algorithm with r and c = 4
        labels, peaks2 = meanshift_opt(feature_matrix, r, c)
        # Converting the resulting cluster centers back to RGB color space
        rgb_peaks = color.lab2rgb(peaks2[:, :3])
        segmented_image = rgb_peaks[labels - 1]  # Assigning colors to each pixel based on the labels
        segmented_image_2 = segmented_image.reshape((n, m, 3))  # Reshaping back to the original image dimensions
        runtime_2 = time.time() - start_time_2 # runtime for optimized algorithm

        return segmented_image_1, segmented_image_2, len(peaks1), len(peaks2), runtime_1, runtime_2

    # run the optimized algorithm
    labels, peaks = meanshift_opt(feature_matrix, r, c)

    # Converting the resulting cluster centers back to RGB color space
    rgb_peaks = color.lab2rgb(peaks[:, :3])

    segmented_image = rgb_peaks[labels - 1]  # Assigning colors to each pixel based on the labels

    segmented_image = segmented_image.reshape((n, m, 3))  # Reshaping back to the original image dimensions



    return segmented_image, len(peaks)

# Implementation on images ends
#----------------------------------------------------------------------------------------------------------------------
# main starts


def main():
    # Start the timer
    start_time = time.time()
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Image Segmentation')

    # Add the command-line arguments
    parser.add_argument('--image', type=str, help='Path to the input image')
    parser.add_argument('-r', type=int, help='Radius parameter')
    parser.add_argument('-c', type=int, help='c in r/c parameter')
    parser.add_argument('--feature_type', type=str, help='Feature type')
    parser.add_argument('--down_size_by', type=int, help='Divisor to down size')
    parser.add_argument('-experiment', type=int, help='Type of experiment')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Extract the argument values
    # python main.py --image <image_path> -r <radius_value> -c <c_value> --feature_type <feature_type_value> --down_size_by <divisor to downsize> -experiment <experiment_type>
    image_path = args.image
    r = args.r
    c = args.c
    feature_type = args.feature_type
    s = args.down_size_by
    exp = args.experiment

    global iterations # to calculate iterations number
    iterations = 0

    # Load the input image
    im = plt.imread(image_path)

    # Downsize the image dividing by s
    downsized_im = resize(im, (im.shape[0] // s, im.shape[1] // s), anti_aliasing=True)
    print("Image size:", downsized_im.shape)

    # Experiment 0: Compare results of simple and optimized algorithm on test data pts.mat
    if exp == 0:
        # simple algorithm
        st_time = time.time()
        test_algo(r)
        runtime = time.time() - st_time
        print(f"Runtime for simple algorithm: {runtime:.2f} seconds")

        # optimized algorithm with speedups
        st_time = time.time()
        test_algo_opt(r)
        runtime = time.time() - st_time
        print(f"Runtime for optimized algorithm: {runtime:.2f} seconds")

    # Experiment 1: Image with simple vs optimized algorithms
    if exp == 1:
        # Perform image segmentation
        segmented_image_1, segmented_image_2, segments_1, segments_2,\
            runtime_1, runtime_2 = segmIm(downsized_im, r, c, exp, feature_type)

        fig, axs = plt.subplots(figsize=(10, 5)) # Figure
        # plot original image
        axs.imshow(downsized_im)
        axs.set_title('Original Image')
        axs.axis('off')
        plt.tight_layout()
        plt.savefig(f"Images\output\exp_1\{image_path[-8:-4]}_0.png", bbox_inches='tight')  # save the image
        plt.show()

        # plot segmented image with simple algorithm
        fig, axs = plt.subplots(figsize=(10, 5))  # Figure
        axs.imshow(segmented_image_1)
        axs.set_title(f'Simple Mean-Shift (r = {r}, c = {c})')
        axs.axis('off')
        axs.text(0, downsized_im.shape[0] + 5, f"Segments: {segments_1}, Runtime: {runtime_1:.2f}s",
                    color='black', fontsize=8, ha='left')
        plt.tight_layout()
        plt.savefig(f"Images\output\exp_1\{image_path[-8:-4]}_1.png", bbox_inches='tight')  # save the image
        plt.show()

        # plot segmented image with optimized algorithm
        fig, axs = plt.subplots(figsize=(10, 5))  # Figure
        axs.imshow(segmented_image_2)
        axs.set_title(f'Optimized Mean-Shift (r = {r}, c = {c})')
        axs.axis('off')
        axs.text(0, downsized_im.shape[0] + 5, f"Segments: {segments_2}, Runtime: {runtime_2:.2f}s",
                    color='black', fontsize=8, ha='left')

        plt.tight_layout()
        plt.savefig(f"Images\output\exp_1\{image_path[-8:-4]}_2.png", bbox_inches='tight')  # save the image
        plt.show()

    # Experiment 2: Effect of different r
    if exp == 2:
        fig, axs = plt.subplots(figsize=(10, 5))  # Figure
        # plot original image
        axs.imshow(downsized_im)
        axs.set_title('Original Image')
        axs.axis('off')
        plt.tight_layout()
        plt.savefig(f"Images\output\exp_2\{image_path[-8:-4]}_0.png", bbox_inches='tight') # save the image
        plt.show()

        for r in range(4, 29, 4):
            start_time = time.time()
            # Perform image segmentation
            segmented_image, segments = segmIm(downsized_im, r, c, exp, feature_type)
            # Calculate the runtime
            runtime = time.time() - start_time
            # plot segmented image with optimized algorithm
            fig, axs = plt.subplots(figsize=(10, 5))  # Figure
            axs.imshow(segmented_image)
            axs.set_title(f'Segmented (r = {r}, c = {c})')
            axs.axis('off')
            axs.text(0, downsized_im.shape[0] + 5, f"Segments: {segments}, Runtime: {runtime:.2f}s",
                        color='black', fontsize=8, ha='left')

            plt.tight_layout()
            plt.savefig(f"Images\output\exp_2\{image_path[-8:-4]}_{r}.png", bbox_inches='tight') # save the image
            plt.show()


    # Experiment 3: Check influence of c on the images
    if exp == 3:
        fig, axs = plt.subplots(figsize=(10, 5))  # Figure
        # plot original image
        axs.imshow(downsized_im)
        axs.set_title('Original Image')
        axs.axis('off')
        plt.tight_layout()
        plt.savefig(f"Images\output\exp_3\{image_path[-8:-4]}_0.png", bbox_inches='tight')  # save the image
        plt.show()

        for c in range(2, 11, 2):
            start_time = time.time()
            # Perform image segmentation
            segmented_image, segments = segmIm(downsized_im, r, c, exp, feature_type)
            # Calculate the runtime
            runtime = time.time() - start_time
            # plot segmented image with optimized algorithm
            fig, axs = plt.subplots(figsize=(10, 5))  # Figure
            axs.imshow(segmented_image)
            axs.set_title(f'Segmented (r = {r}, c = {c})')
            axs.axis('off')
            axs.text(0, downsized_im.shape[0] + 5, f"Segments: {segments}, Runtime: {runtime:.2f}s",
                     color='black', fontsize=8, ha='left')

            plt.tight_layout()
            plt.savefig(f"Images\output\exp_3\{image_path[-8:-4]}_{c}.png", bbox_inches='tight')  # save the image
            plt.show()


    # Experiment 4: 3D vs 5D feature space with varying r
    if exp == 4:
        fig, axs = plt.subplots(figsize=(10, 5))  # Figure
        # plot original image
        axs.imshow(downsized_im)
        axs.set_title('Original Image')
        axs.axis('off')
        plt.tight_layout()
        plt.savefig(f"Images\output\exp_4\{image_path[-8:-4]}_0.png", bbox_inches='tight')  # save the image
        #plt.show()

        for r in range(20, 45, 4):
            start_time = time.time()
            # Perform image segmentation
            segmented_image, segments = segmIm(downsized_im, r, c, exp, feature_type)
            # Calculate the runtime
            runtime = time.time() - start_time
            # plot segmented image with optimized algorithm
            fig, axs = plt.subplots(figsize=(10, 5))  # Figure
            axs.imshow(segmented_image)
            axs.set_title(f'Segmented (r = {r}, c = {c})')
            axs.axis('off')
            axs.text(0, downsized_im.shape[0] + 5, f"Segments: {segments}, Runtime: {runtime:.2f}s",
                     color='black', fontsize=8, ha='left')

            plt.tight_layout()
            plt.savefig(f"Images\output\exp_4\{image_path[-8:-4]}_{r}.png", bbox_inches='tight')  # save the image
            #plt.show()


if __name__ == '__main__':
    main()
