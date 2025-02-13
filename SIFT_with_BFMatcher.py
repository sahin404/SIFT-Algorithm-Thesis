# Techniques Used in This Code:
# ✅ Feature Detection: Used SIFT (Scale-Invariant Feature Transform) to extract keypoints and descriptors.
# ✅ Feature Matching: Used BFMatcher (Brute Force Matcher) to match features between two images.
# ✅ Matching Accuracy: Calculated the percentage of correct matches based on the number of matched features.
# TODO: Only change the Images path

import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # Import for displaying images in Colab

# Load two images in grayscale
img1 = cv2.imread('/content/angle1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/content/angle2.jpg', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded properly
if img1 is None or img2 is None:
    print("Error: One or both images not found. Check the file paths!")
else:
    # Create SIFT feature detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Print total detected features
    print(f"Total features detected in Original image: {len(keypoints1)}")
    print(f"Total features detected in Edited image: {len(keypoints2)}")

    # Create BFMatcher (Brute Force Matcher) with L2 norm
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match features between the two images
    matches = bf.match(descriptors1, descriptors2)


    # Print total matched features
    print(f"Total matched features: {len(matches)}")

    #Calculate percentage of correct matches (total matches / total features)
    correct_matches_percentage = len(matches) / min(len(keypoints1), len(keypoints2)) * 100
    print(f"Percentage of correct matches in First Code: {correct_matches_percentage:.2f}%")

    # Draw all matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matched image (Colab-Compatible)
    cv2_imshow(img_matches)
