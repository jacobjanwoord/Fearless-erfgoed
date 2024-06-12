import cv2
import numpy as np
from matplotlib import pyplot as plt

def sift_similarity(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0.0

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return 0.0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
    matches_mask = mask.ravel().tolist()

    num_inliers = np.sum(matches_mask)
    num_matches = len(good_matches)
    
    similarity_percentage = (num_inliers / num_matches) * 100
    return similarity_percentage

# Load two grayscale images
img1 = cv2.imread('nk_collection_meubels_cleaned/meubel_5.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('nk_collection_meubels_cleaned/meubel_6.jpg', cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Could not open or find the images!")
else:
    num_good_matches = sift_similarity(img1, img2)
    print(f"Percentage score: {num_good_matches}")