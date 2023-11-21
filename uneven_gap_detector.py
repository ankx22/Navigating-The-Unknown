# from skimage import io, filters, measure, morphology, color
# from skimage.measure import regionprops, label
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2

# # Helper function to calculate the area of a contour
# def contour_area(contour):
#     # contour is expected to be an (N, 2) array where the first column is y coordinates
#     # and the second column is x coordinates. The area is then computed using the
#     # shoelace formula.
#     x = contour[:, 1]
#     y = contour[:, 0]
#     return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# # Load the image
# image_path = r"D:\WPI\Sem 3\Aerial vehicles\HW3\outputs\001.png"
# image = io.imread(image_path)

# # If the image has an alpha channel, remove it
# if image.shape[-1] == 4:
#     image = color.rgba2rgb(image)

# # Convert image to grayscale
# gray_image = color.rgb2gray(image)

# # Step 1: Noise reduction with Gaussian blur
# blurred_image = filters.gaussian(gray_image, sigma=1)

# # Step 2: Apply Otsu's method to perform thresholding
# thresh = filters.threshold_otsu(blurred_image)
# binary_image = blurred_image > thresh

# # Step 3: Morphological operations to remove small objects (minor gaps)
# # Since we're interested in larger gaps, we can use opening to remove small objects
# cleaned_image = morphology.remove_small_objects(binary_image, min_size=500)

# # Step 4: Find contours of the remaining objects (gaps)
# contours = measure.find_contours(cleaned_image, 0.8)
# print("shape of cotours",len(contours))
# # Step 5: Calculate the area of each contour to find the largest one
# largest_area = 0
# largest_contour = None
# for contour in contours:
#     area = contour_area(contour)
#     if area > largest_area:
#         largest_area = area
#         largest_contour = contour

# largest_contour_cv = np.array(largest_contour, dtype=np.int32).reshape((-1, 1, 2))

# binary_gap = np.zeros(gray_image.shape, dtype=np.uint8)

# # Fill in the largest gap in the binary image
# for coord in largest_contour:
#     binary_gap[int(coord[0]), int(coord[1])] = 1

# # Label the regions in the binary image
# labeled_gap = label(binary_gap)

# # Calculate the properties of the labeled regions
# props = regionprops(labeled_gap)

# # Get the centroid of the largest gap (there should only be one labeled region)
# gap_centroid = props[0].centroid if props else (None, None)

# # M = cv2.moments(largest_contour_cv)
# # if M["m00"] != 0:
# #     cX = int(M["m10"] / M["m00"])
# #     cY = int(M["m01"] / M["m00"])
# #     print("The centroid of the gap detected is:",cX,",",cY)
# # else:
# #     cX, cY = 0, 0  # Can't compute centroid because the contour area is zero
# #     print(" Cant compute centroid as contour area is zero.")

# print("The centroid of the gap detected is:",int(gap_centroid[0]),",",int(gap_centroid[1]))
# (h, w) = image.shape[:2]
# print("The centroid of the image frame is:",h//2,",",w//2)

# # # Let's plot the original image and the largest gap detected
# # fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# # # Original image
# # ax[0].imshow(image, cmap='gray')
# # ax[0].set_title('Original Image')
# # ax[0].axis('off')


# x = None
# y = None
# cleaned_image = (cleaned_image * 255).astype(np.uint8)
# cleaned_image = cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2RGB)
# print(cleaned_image.dtype)
# largest_contour = largest_contour.reshape((-1, 1, 2)).astype(np.int32)
# print("shape of cotours",largest_contour.shape)

# # Binary image with the largest gap highlighted
# # ax[1].imshow(cleaned_image, cmap='gray')
# if largest_contour is not None:
#     x = gap_centroid[1]
#     y = gap_centroid[0]
#     print(f"centers are {x,y}")
#     cv2.circle(cleaned_image, (int(x), int(y)), 7, (0, 0, 255), -1)
#     print("added center")
#     # print(largest_contour[:,1])
#     cv2.drawContours(cleaned_image, largest_contour, -1, (0,255,0), 3)
#     # cv2.drawContours(cleaned_image, largest_contour[:, 0], -1, (0,255,0), 3)
#     # ax[1].plot(largest_contour[:, 1], largest_contour[:, 0], linewidth=3)
#     # ax[1].plot(cX, cY, 'ro')  # Centroid in red
#     # ax[1].plot( 'ro')  # Centroid in red
# print(f"hole center is {x,y}")
# filepath = f"D:\WPI\Sem 3\Aerial vehicles\HW3\contour_w_center\ frame.png"
# cv2.imwrite(filepath,cleaned_image)



# # Binary image with the largest gap highlighted
# ax[1].imshow(cleaned_image, cmap='gray')
# if largest_contour is not None:
#     ax[1].plot(largest_contour[:, 1], largest_contour[:, 0], linewidth=3)
#     # ax[1].plot(cX, cY, 'ro')  # Centroid in red
#     ax[1].plot(gap_centroid[1], gap_centroid[0], 'ro')  # Centroid in red

#     # # Since OpenCV expects integer points, we need to convert them
#     # largest_contour_int = np.round(largest_contour).astype(int)
#     # cv2.drawContours(cleaned_image, [largest_contour_int], -1, (0, 255, 0), 3)

#     # # Draw the centroid
#     # centroid_int = (int(gap_centroid[1]), int(gap_centroid[0]))
#     # cv2.circle(cleaned_image, centroid_int, 7, (0, 0, 255), -1)

# # # Save the image with the drawn contour and centroid
# # output_path = 'output_with_contour_and_centroid.png'
# # cv2.imwrite(output_path, cleaned_image)


# ax[1].set_title('Largest Gap Detected')
# ax[1].axis('off')

# plt.tight_layout()

# # Save the figure to a file
# plt.savefig('output_image_with_contours.png')
# plt.show()

import cv2
import numpy as np

# Helper function to calculate the area of a contour
def contour_area(contour):
    return cv2.contourArea(contour)

# Load the image
image_path = r"D:\WPI\Sem 3\Aerial vehicles\HW3\outputs\001.png"
image = cv2.imread(image_path)

# If the image has an alpha channel, remove it
if image.shape[-1] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Noise reduction with Gaussian blur
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)

# Step 2: Apply Otsu's method to perform thresholding
ret, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 3: Morphological operations to remove small objects (minor gaps)
# Define the structuring element
kernel = np.ones((3, 3), np.uint8)
cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 4: Find contours of the remaining objects (gaps)
contours, hierarchy = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 5: Calculate the area of each contour to find the largest one
largest_area = 0
largest_contour = None
for contour in contours:
    area = contour_area(contour)
    if area > largest_area:
        largest_area = area
        largest_contour = contour
cleaned_image = (cleaned_image * 255).astype(np.uint8)
cleaned_image = cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2RGB)
# Draw the largest contour and centroid if it exists
if largest_contour is not None:
    # Draw the largest contour
    cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 3)

    # Calculate the centroid of the contour
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(image, (cX, cY), 7, (0, 0, 255), -1)
        print("The centroid of the gap detected is:", cX, ",", cY)
    else:
        print("Can't compute centroid as contour area is zero.")

# Save the image with the drawn contour and centroid
output_path = r"D:\WPI\Sem 3\Aerial vehicles\HW3\contour_w_center\frame.png"

cv2.imwrite(output_path, cleaned_image)

print(f"Image saved at {output_path}")
