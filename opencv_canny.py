import cv2
import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.filters import threshold_otsu

def canny_edge_detection(image_path, low_threshold, high_threshold):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)

    return edges

def calculate_lower_threshold(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Use Otsu's method to calculate the optimal threshold value
    lower_threshold = threshold_otsu(gray_image)

    return lower_threshold

def calculate_upper_threshold(lower_threshold, multiplier=900):
    # Adjust the upper threshold as a multiple of the lower threshold
    upper_threshold = lower_threshold * multiplier

    return upper_threshold

if __name__ == '__main__':
    input_image_path = "/home/hasan/Desktop/Canny/broken_bones_f.jpeg"
    
    # Load the image using skimage.io
    image = io.imread(input_image_path, as_gray=True)

    # Calculate lower and upper thresholds
    lower_threshold = calculate_lower_threshold(image)
    upper_threshold = calculate_upper_threshold(lower_threshold)

    print("Lower Threshold:", lower_threshold)
    print("Upper Threshold:", upper_threshold)

    edges = canny_edge_detection(input_image_path,5, lower_threshold, upper_threshold)

    # Plot the original image and the edges
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny Edge Detection")
    plt.axis('off')

    plt.show()
