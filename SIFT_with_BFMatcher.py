# Techniques Used in This Code:
# ✅ Feature Detection: Used SIFT (Scale-Invariant Feature Transform) to extract keypoints and descriptors.
# ✅ Feature Matching: Used BFMatcher (Brute Force Matcher) to match features between two images.
# ✅ Matching Accuracy: Calculated the percentage of correct matches based on the number of matched features.
# ✅ knnMatch: Find two best matches for each descriptor
# ✅ Apply Lowe’s ratio test (0.75 threshold)
# TODO: Only change the Images path

#Raw SIFT
import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # For displaying images in Colab

# Load images in grayscale
img1 = cv2.imread('/content/angle4.jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/content/angle6_edit.jpg', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded properly
if img1 is None or img2 is None:
    print("Error: One or both images not found. Check the file paths!")
else:
    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    print(f"Total keypoints in Image 1: {len(keypoints1)}")
    print(f"Total keypoints in Image 2: {len(keypoints2)}")

    # Use BFMatcher with L2 norm
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Find two best matches for each descriptor
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe’s ratio test (0.75 threshold)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    print(f"Total good matches: {len(good_matches)}")

    # Calculate percentage of correct matches
    correct_matches_percentage = len(good_matches) / min(len(keypoints1), len(keypoints2)) * 100
    print(f"Percentage of correct matches: {correct_matches_percentage:.2f}%")

    # Draw only good matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matched image (Colab-Compatible)
    cv2_imshow(img_matches)
