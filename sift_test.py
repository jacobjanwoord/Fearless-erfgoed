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


def compute_similarities(nk_img_path, munich_imgs, path):
    """
    This function takes three arguments: 
    - nk_img, which is a single image from the nk collection. 
    - munich_imgs, this contains all images from the Munich Database. 
    - path, this is the path to the gray scaled Munich Database.
    
    It then computes the feature descriptor for the nk collection image and all the images in the \
    Munich Database. Afterwards takes the dot-product to get the dot-product similiarity. It then \
    saves the similarity and the two images as key-value pairs in a dictionary. 
    """
    
    nk_img = cv2.imread(nk_img_path, cv2.IMREAD_GRAYSCALE)
    similarities = {}
    i = 0
    for img_name in munich_imgs:
        print(i)
        i += 1
        img_path = os.path.join(path, img_name)
        munich_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if nk_img is not None and munich_img is not None:
            similarity = sift_similarity(nk_img, munich_img)
            nk_img_name = os.path.basename(nk_img_path)
            munich_img_name = os.path.basename(img_path)
            similarities[(nk_img_name, munich_img_name)] = similarity

    return similarities

# Example usage
nk_img = "test_dataset_gray/kast_nk.jpg"
munich_imgs = os.listdir("scraped_images_grayscaled_big")
path = "scraped_images_grayscaled_big"

similarities = compute_similarities(nk_img, munich_imgs, path)
print(similarities)

def visualize_keypoints(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("The image path is invalid.")

    # Initialize the AKAZE detector
    akaze = cv2.AKAZE_create()

    # Detect keypoints and descriptors with AKAZE
    kp, des = akaze.detectAndCompute(img, None)

    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0), flags=cv2.DrawMatchesFlags_DEFAULT)

    # Display the image with keypoints
    plt.imshow(img_with_keypoints, cmap='gray')
    plt.title(f'Image: Keypoints')
    plt.axis('off')
    plt.show()


# Example usage:
image_path = 'test_dataset_gray/tafel_nk.jpg'


img1 = cv2.imread('test_dataset_gray/speeltafel_nk.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('test_dataset_gray/tafel_mccp.jpg', cv2.IMREAD_GRAYSCALE)
num_good_matches = sift_similarity(img1, img2)
print(f"Similarity percentage:{num_good_matches}")

compute_similarities(nk_img, munich_imgs, path)
