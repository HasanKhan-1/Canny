import canny_solver
import opencv_canny
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import threshold_otsu

def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def get_optimal_thresholds(image):
    # Calculate the optimal threshold using Otsu's method
    optimal_threshold = threshold_otsu(image)

    # Set the lower threshold to 0 (minimum intensity value)
    lower_threshold = 0

    # Set the upper threshold to the calculated optimal threshold value
    upper_threshold = optimal_threshold

    return lower_threshold, upper_threshold


if __name__ == '__main__':
    input_image_path = "/home/hasan/Desktop/Canny/broken_bones_f.jpeg"
    input_image_path_trans = "/home/hasan/Desktop/Canny/broken_bones.jpeg"

    image = io.imread(input_image_path, as_gray=True)

    lower_threshold_custom, upper_threshold_custom = get_optimal_thresholds(image)
    lower_threshold_cv, upper_threshold_cv = get_optimal_thresholds(image)

    # Get edges using OpenCV's Canny for image 1
    edges_opencv_a = opencv_canny.canny_edge_detection(input_image_path, 30, 90)
    edges_opencv_b = opencv_canny.canny_edge_detection(input_image_path, 70, 90)
    edges_opencv_c = opencv_canny.canny_edge_detection(input_image_path, 60, 140)
    edges_opencv_d = opencv_canny.canny_edge_detection(input_image_path, 65, 120)

    edges_your_a = canny_solver.canny_edge_detector(input_image_path, 5, 30, 90)
    edges_your_b = canny_solver.canny_edge_detector(input_image_path, 5, 90, 120)
    edges_your_c = canny_solver.canny_edge_detector(input_image_path, 5, 120, 140)
    edges_your_d = canny_solver.canny_edge_detector(input_image_path, 5, 95, 110)

    # Get edges using OpenCV's Canny for image 1

    edges_opencv_e = opencv_canny.canny_edge_detection(input_image_path_trans, 30, 90)
    edges_opencv_f = opencv_canny.canny_edge_detection(input_image_path_trans, 70, 140)
    edges_opencv_g = opencv_canny.canny_edge_detection(input_image_path_trans, 175, 200)
    edges_opencv_h = opencv_canny.canny_edge_detection(input_image_path_trans, 220, 255)

    edges_your_e = canny_solver.canny_edge_detector(input_image_path_trans, 5, 30, 90)
    edges_your_f = canny_solver.canny_edge_detector(input_image_path_trans, 5, 90, 120)
    edges_your_g = canny_solver.canny_edge_detector(input_image_path_trans, 5, 120, 140)
    edges_your_h = canny_solver.canny_edge_detector(input_image_path_trans, 5, 170, 255)



    # Get edges using your Canny
    edges_your = canny_solver.canny_edge_detector(input_image_path, 5, 30, 90)

    # Calculate accuracy using Mean Squared Error (MSE)
    accuracy = 1.0 / (1.0 + calculate_mse(edges_your_d, edges_opencv_d))
    print(accuracy)

    accuracy1 = 1.0 / (1.0 + calculate_mse(edges_opencv_h, edges_your_h))
    print(accuracy1)

    #print("Accuracy: {:.2f}%".format(accuracy * 100))

    # Plot the original image, OpenCV's Canny edges, and Your Canny edges
    plt.figure(figsize=(12, 6))

    # Original Image
    # plt.subplot(1, 3, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title("Original Image")
    # plt.axis('off')

    # # OpenCV's Canny Edge Detection
    # plt.subplot(1, 3, 2)
    # plt.imshow(edges_opencv, cmap='gray')
    # plt.title("OpenCV's Canny Edge Detection")
    # plt.axis('off')

    # # Your Canny Edge Detection
    # plt.subplot(1, 3, 3)
    # plt.imshow(edges_your, cmap='gray')
    # plt.title("Your Canny Edge Detection")
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()

#OpenCV implementation (finding the right thresholds) Compound Break
    # plt.subplot(2, 2, 1)
    # plt.imshow(edges_opencv_a, cmap='gray')
    # plt.title("a) Low: 30 High: 90")
    # plt.axis('off')

    # plt.subplot(2, 2, 2)
    # plt.imshow(edges_opencv_b, cmap='gray')
    # plt.title("b) Low: 70 High: 120")
    # plt.axis('off')

    # plt.subplot(2, 2, 3)
    # plt.imshow(edges_opencv_c, cmap='gray')
    # plt.title("c) Low: 60 High: 140")
    # plt.axis('off')

    # plt.subplot(2, 2, 4)
    # plt.imshow(edges_opencv_d, cmap='gray')
    # plt.title("d) Low: 50 High: 120")
    # plt.axis('off')

#My implementation (finding the right thresholds) Compound Break
    # plt.subplot(2, 2, 1)
    # plt.imshow(edges_your_a, cmap='gray')
    # plt.title("a) Low: 30 High: 90")
    # plt.axis('off')

    # plt.subplot(2, 2, 2)
    # plt.imshow(edges_your_b, cmap='gray')
    # plt.title("b) Low: 90 High: 120")
    # plt.axis('off')

    # plt.subplot(2, 2, 3)
    # plt.imshow(edges_your_c, cmap='gray')
    # plt.title("c) Low: 120 High: 140")
    # plt.axis('off')

    # plt.subplot(2, 2, 4)
    # plt.imshow(edges_your_d, cmap='gray')
    # plt.title("d) Low: 95 High: 110")
    # plt.axis('off')


    # plt.tight_layout()
    # plt.show()

#My implementation (finding the right thresholds) Transverse Break
    # plt.subplot(2, 2, 1)
    # plt.imshow(edges_your_e, cmap='gray')
    # plt.title("a) Low: 30 High: 90")
    # plt.axis('off')

    # plt.subplot(2, 2, 2)
    # plt.imshow(edges_your_f, cmap='gray')
    # plt.title("b) Low: 90 High: 120")
    # plt.axis('off')

    # plt.subplot(2, 2, 3)
    # plt.imshow(edges_your_g, cmap='gray')
    # plt.title("c) Low: 120 High: 140")
    # plt.axis('off')

    # plt.subplot(2, 2, 4)
    # plt.imshow(edges_your_h, cmap='gray')
    # plt.title("d) Low: 170 High: 255")
    # plt.axis('off')


    # plt.tight_layout()
    # plt.show()

#OPenCV implementation (finding the right thresholds) Transverse Break
    plt.subplot(2, 2, 1)
    plt.imshow(edges_opencv_e, cmap='gray')
    plt.title("a) Low: 30 High: 90")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(edges_opencv_f, cmap='gray')
    plt.title("b) Low: 90 High: 120")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(edges_opencv_g, cmap='gray')
    plt.title("c) Low: 120 High: 140")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(edges_opencv_h, cmap='gray')
    plt.title("d) Low: 170 High: 255")
    plt.axis('off')


    plt.tight_layout()
    plt.show()

    