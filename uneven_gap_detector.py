import cv2
import numpy as np

# Helper function to calculate the area of a contour
def contour_area(contour):
    return cv2.contourArea(contour)

# Load the image
image_path = r"/home/pear/Downloads/Navigating-The-Unknown/flow/frame005.png"
image = cv2.imread(image_path)

# If the image has an alpha channel, remove it
if image.shape[-1] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Noise reduction with Gaussian blur
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)

# Step 2: Apply Otsu's method to perform thresholding
# ret, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Use adaptive thresholding to get a binary image
adaptive_thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 19, 2)

# Use Canny edge detection
edges = cv2.Canny(adaptive_thresh, 50, 150)

# Dilate the edges to close the gaps
dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

# Apply closing to fill in gaps
closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

# # Step 3: Morphological operations to remove small objects (minor gaps)

cv2.imshow('binary_image',closed_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 4: Find contours of the remaining objects (gaps)
contours, hierarchy = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Sort contours by area and get the largest one
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

# Draw the largest contour and centroid if it exists
for contour in contours:
    if cv2.contourArea(contour) > 100:  # Threshold to filter small contours
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
            print("The centroid of the largest contour detected is:", cX, ",", cY)

# Save the image with the drawn contour and centroid
output_path = r"/home/pear/Downloads/Navigating-The-Unknown/frame.png"

cv2.imwrite(output_path, image)


print(f"Image saved at {output_path}")
