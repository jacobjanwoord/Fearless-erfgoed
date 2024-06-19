import cv2
import os
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

    if len(good_matches) < 10:
        return 0.0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
    matches_mask = mask.ravel().tolist()

    num_inliers = np.sum(matches_mask)
    num_matches = len(good_matches)
    
    similarity_percentage = (num_inliers / num_matches) * 100
    return similarity_percentage

image_folder = 'test_dataset_gray'

# List to hold the images
images_nk = []
images_mccp = []

# Loop through the files in the directory
for filename in os.listdir(image_folder):
    file_path = os.path.join(image_folder, filename)
    
    # Read the image file using OpenCV
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if 'nk' in filename:
        images_nk.append((filename, img))
    if 'mccp' in filename:
        images_mccp.append((filename, img))

for filename_nk, image_nk in images_nk:
    for filename_mccp, image_mccp in images_mccp:
        num_good_matches = sift_similarity(image_nk, image_mccp)
        print(f"Similarity percentage:{filename_nk, filename_mccp} {num_good_matches}")