# RoadLineDetection
Implement algorithm for detecting road lines from image
## Steps
Algorithm can be split into several steps
1. Convert image to grayscale
2. Apply Canny Edge Detection
    a. Apply Gaussian filter to remove additional noise
    b. Calculate intensity gradients
    c. Apply Non-Maximum Supression
    d. Double threshold
    e. Apply hysteresis 
3. Use Hough Transform to detect Lines
4. Draw lines on input image