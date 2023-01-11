import numpy as np


def conv2d(image: np.array, kernel: np.array) -> np.array:

    # Get the kernel dimensions
    kernel_height, kernel_width = kernel.shape

    # Calculate padding size
    pad_size_h = (kernel_height - 1) // 2
    pad_size_w = (kernel_width - 1) // 2

    # Add padding to the input image
    image = np.pad(image, ((pad_size_h, pad_size_h),
                   (pad_size_w, pad_size_w)), mode='constant')

    # Get the image and kernel dimensions
    image_height, image_width = image.shape

    # Compute the dimensions of the output image
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # Create a numpy array for the output image
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            subregion = image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = (subregion*kernel).sum()

    return output


def create_gaussian_kernel(size: np.int32, sigma: np.float32) -> np.array:
    size = size // 2
    x, y = np.mgrid[-size: size+1, -size: size+1]
    norm = 1 / (2.0 * np.pi * sigma**2)
    kernel = np.exp(-((x**2 - y**2)/(2*sigma**2)))*norm

    return kernel


def gaussian_blur(image: np.array, kernel_size: tuple, sigma: float) -> np.array:
    # Make sure kernel size is squared
    assert kernel_size[0] == kernel_size[1]

    # Generate gaussian kernel
    kernel = create_gaussian_kernel(kernel_size[0], sigma)

    # Apply Gaussian kernel on image
    blurred_image = conv2d(image, kernel)

    return blurred_image


def calculate_gradient_intensity(image: np.array) -> tuple((np.array, np.array)):
    # Sobel-Feldman kernels
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    i_x = conv2d(image, kernel_x)
    i_y = conv2d(image, kernel_y)

    g = (i_x**2 + i_y**2)**(1/2)
    g = g/g.max()*255

    theta = np.arctan2(i_y, i_x)

    return (g, theta)


def apply_non_max_suppresion(image: np.array, theta: np.array) -> np.array:
    height, width = image.shape
    output = np.zeros(image.shape, dtype=np.int32)
    angle = theta*100 / np.pi
    angle[angle < 0] += 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            try:
                q = 255
                r = 255

               # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = image[i, j+1]
                    r = image[i, j-1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = image[i+1, j-1]
                    r = image[i-1, j+1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = image[i+1, j]
                    r = image[i-1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = image[i-1, j-1]
                    r = image[i+1, j+1]

                if (image[i, j] >= q) and (image[i, j] >= r):
                    output[i, j] = image[i, j]
                else:
                    output[i, j] = 0

            except IndexError as e:
                pass
    return output


def double_threshold(image: np.array, low_threshold: np.int32, high_threshold: np.int32) -> tuple((np.array, np.int32, np.int32)):

    result = np.zeros(image.shape, dtype=np.int32)

    weak = np.int32(75)
    strong = np.int32(255)

    weak_i, weak_j = np.where((image >= low_threshold)
                              & (image < high_threshold))
    strong_i, strong_j = np.where(image >= high_threshold)

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return (result, weak, strong)


def hysteresis(image, weak, strong) -> np.array:
    # Apply Conv2D to get active signals where there is at least single strong pixel around
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)

    mask = np.zeros(image.shape)
    mask[image == strong] = 1
    out_mask = conv2d(mask, kernel)

    # Set all pixels to strong if at least 1 neighbor is strong
    image[(out_mask > 0) & (image == weak)] = strong
    image[image == weak] = 0
    return image


def detect_edges(gray_img: np.array, low_threshold: np.int32, high_threshold: np.int32) -> np.array:
    # Implementation of canny edge detection

    # Noise reduction
    denoised_img = gaussian_blur(gray_img, (5, 5), 1)
    # Intensity Gradient calculation
    gradients, theta = calculate_gradient_intensity(denoised_img)
    # Non-maximum supressionx
    img = apply_non_max_suppresion(gradients, theta)
    # Double Threshold
    thresholded_img, weak, strong = double_threshold(
        img, low_threshold, high_threshold)
    # Hysteresis Thresholding
    img = hysteresis(thresholded_img, weak, strong)

    return img
