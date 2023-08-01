'''
The canny filter operates by the following steps:
    1.First we typically convert our image to greyscale
    2. Noise Reduction-> Pre-Processing
        - Done by utilizing gaussian_blur
    3. Gradient Calculation
    4. Non-maximum reduction
    5. Double Threshold
    6. Edge Tracking by Hysteresis
'''


#Import required libraries
import numpy as np #This contains math that is need for our calculations 
import cv2
import matplotlib.pyplot as plt #Serves as the tool we will use for plotting our functions
from skimage.filters import threshold_otsu # This library is used to determine the best threshold values for the image

'''
Step 1: Getting the ability to access our image. 
    In this case the image was loaded and displayed by utilizing opencv and pyplot
'''
def loading_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Error loading the image. Please check the image path.")
    return img

def displaying_image(image):
    plt.imshow(image,cmap="gray")
    plt.axis('off')
    plt.show()

#We addititionally have grey scaling which makes the picture easier to read 
def grey_scale(image):
    greyed_image = np.dot(image[..., :3], [0.299,0.587,0.114]) 
    #image[..., :3]: Gets all the 
    return greyed_image


'''
Step 2: Pre-processing (Noise Reduction)
    This includes applying the Gaussian Filter
'''

# def gaussian_filter(image, kernel_size):
#     #Here we define the 2D Gaussian filter 
#     kernel = np.outer

#     return reduced_noise_img

import numpy as np

def convolution(image, kernel):
    # Get the dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the padding required for 'same' convolution
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Create a zero-padded image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Initialize the output image
    output_image = np.zeros_like(image)

    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            output_image[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output_image

def gaussian_filter(image, kernel_size, sigma=1.0):
    # Define the nD Gaussian kernel
    kernel_range = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-(kernel_range ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    # Apply the 1D Gaussian kernel along the height of the image
    blurred_height = convolution(image, kernel.reshape(-1, 1))

    # Apply the 1D Gaussian kernel along the width of the image
    blurred = convolution(blurred_height, kernel.reshape(1, -1))

    return blurred

def calculate_gradients(image):
    # Define Sobel operators
    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Perform convolution to get x and y gradients
    sobel_x = convolution(image, sobel_x_kernel)
    sobel_y = convolution(image, sobel_y_kernel)

    # Calculate gradient magnitude and direction
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2) #Just Pythagoream Theorum 
    gradient_direction = np.arctan2(sobel_y, sobel_x) #How we would regularly get angle

    return gradient_magnitude, gradient_direction    

def non_max_suppression(gradient_magnitude, gradient_direction):
    suppressed = np.zeros_like(gradient_magnitude) #This matrix will hold the suppressed information that we create 
    angle = gradient_direction * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]

            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                suppressed[i, j] = gradient_magnitude[i, j]
            else:
                suppressed[i, j] = 0

    return suppressed


def hysteresis_thresholding(image, low_threshold, high_threshold):
    high_value = 255
    low_value = 100

    strong_edges = (image >= high_threshold)
    weak_edges = (image >= low_threshold) & (image < high_threshold)

    edges = np.zeros_like(image)
    edges[strong_edges] = high_value
    edges[weak_edges] = low_value

    return edges

def get_optimal_thresholds(image):
    # Convert the image to grayscale
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the optimal threshold using Otsu's method
    optimal_threshold = threshold_otsu(grey_image)

    # Set the lower threshold to 0 (minimum intensity value)
    lower_threshold = 0

    # Set the upper threshold to the calculated optimal threshold value
    upper_threshold = optimal_threshold

    return lower_threshold, upper_threshold


#We create a function so that it is less convoluted when calling in main
def canny_edge_detector(image_path, kernel_size, low_threshold, high_threshold):
    #Step 1
    original_image = loading_image(image_path)
    grey_image = grey_scale(original_image)

    #Step 2
    gaussian_filter_img = gaussian_filter(grey_image, 5)

    #Step 3: 
    gradient_magnitude, gradient_direction= calculate_gradients(gaussian_filter_img)

    #Step 4:
    suppressed_edges = non_max_suppression(gradient_magnitude, gradient_direction)

    #Step 5:
    final_edges = hysteresis_thresholding(suppressed_edges, low_threshold, high_threshold)

    return final_edges


if __name__=='__main__':
    input_image_path = "/home/hasan/Desktop/Canny/broken_bones_f.jpeg"
    image = cv2.imread(input_image_path)
   
    finished_img = canny_edge_detector(input_image_path, 5, 95,110)
    finished_img_7 = canny_edge_detector(input_image_path, 7, 95,110)
    finished_img_3 = canny_edge_detector(input_image_path, 3, 95,110)

    plt.figure(figsize=(12, 4))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(finished_img_3, cmap='gray')
    plt.title("3x3 kernel")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(finished_img, cmap='gray')
    plt.title("5x5 kernel")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(finished_img_7, cmap='gray')
    plt.title("7x7 kernel")
    plt.axis('off')

    plt.show()